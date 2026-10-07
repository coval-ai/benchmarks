# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, S608
"""Exercise the cleanup and its rollback against published PostgreSQL data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pytest_postgresql.factories import postgresql

pg_conn = postgresql("pg_proc")
COMPAT = "20261005_0043"
CLEANUP = "20261007_0044"
TABLES = (
    "metric_evaluations",
    "dashboard_metric_values",
    "metric_values_by_bucket",
    "dashboard_hourly_aggregates",
)
WINDOWS = ("24h", "7d", "30d")


def _migrate(conn: Any, revision: str, *, opt_in: bool = False, down: bool = False) -> None:
    conn.commit()
    cfg = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    info = conn.info
    cfg.set_main_option(
        "sqlalchemy.url", f"postgresql+psycopg://{info.user}@{info.host}:{info.port}/{info.dbname}"
    )
    if opt_in:
        cfg.attributes["allow_metric_code_cleanup"] = True
    (command.downgrade if down else command.upgrade)(cfg, revision)


@pytest.fixture
def compat(pg_conn: Any) -> Any:
    _migrate(pg_conn, COMPAT)
    pg_conn.autocommit = True
    return pg_conn


def _seed(conn: Any, *, sample: str = "first") -> None:
    with conn.transaction():
        run = conn.execute(
            """INSERT INTO benchmarks_v2.runs
          (dataset_id,dataset_sha256,runner_sha,status,finished_at,scheduled_at)
          VALUES ('d',%s,'test','succeeded','2026-10-01 12:00Z','2026-10-01 11:00Z') RETURNING id""",
            ("a" * 64,),
        ).fetchone()[0]
        for i in range(2):
            observation = conn.execute(
                """INSERT INTO benchmarks_v2.benchmark_observations
              (run_id,dataset_id,dataset_sha256,sample_id,provider,model,benchmark,source_kind,status,captured_at)
              VALUES (%s,'d',%s,%s,'p','m','S2S','dataset_audio','succeeded','2026-10-01 11:50Z') RETURNING id""",
                (run, "b" * 64, f"{sample}-{i}"),
            ).fetchone()[0]
            for code in ("WER", "TTFA"):
                evaluation = conn.execute(
                    """INSERT INTO benchmarks_v2.metric_evaluations
                  (observation_id,metric_type,metric_version,evaluation_variant,executor,status)
                  VALUES (%s,%s,'v1','default','test','queued') RETURNING id""",
                    (observation, code),
                ).fetchone()[0]
                conn.execute(
                    "UPDATE benchmarks_v2.metric_evaluations SET status='running',started_at=now() WHERE id=%s",
                    (evaluation,),
                )
                values = (
                    {
                        "primary": ("percent", 50 if i == 0 else 5),
                        "substitution_count": ("count", 1),
                        "deletion_count": ("count", i),
                        "insertion_count": ("count", 0),
                        "reference_words": ("count", 2 if i == 0 else 40),
                        "insertions": ("percent", 0),
                        "deletions": ("percent", 0 if i == 0 else 2.5),
                        "substitutions": ("percent", 50 if i == 0 else 2.5),
                    }
                    if code == "WER"
                    else {
                        "primary": ("ms", 100 + 100 * i),
                        "roundtrip": ("ms", 20 + 30 * i),
                        "leading_silence": ("ms", 10 + 5 * i),
                    }
                )
                for key, (unit, value) in values.items():
                    conn.execute(
                        """INSERT INTO benchmarks_v2.metric_values
                      (metric_evaluation_id,value_key,unit,value,value_role) VALUES (%s,%s,%s,%s,%s)""",
                        (
                            evaluation,
                            key,
                            unit,
                            value,
                            "primary" if key == "primary" else "component",
                        ),
                    )
                conn.execute(
                    "UPDATE benchmarks_v2.metric_evaluations SET status='succeeded',finished_at=now() WHERE id=%s",
                    (evaluation,),
                )
        conn.execute("""INSERT INTO benchmarks_v2.metric_values_by_bucket
          (provider,model,benchmark,dataset_id,metric_type,metric_version,evaluation_variant,value_key,unit,bucket_at,min_value,p25,p50,p75,max_value,value_sum,sample_count)
          VALUES ('p','m','S2S','d','WER','v1','default','primary','percent','2026-10-01 11:00Z',5,5,5,5,5,5,1)
          ON CONFLICT DO NOTHING""")
        conn.execute("""INSERT INTO benchmarks_v2.dashboard_hourly_aggregates
          (provider,model,benchmark,dataset_id,metric_type,metric_version,evaluation_variant,hour_at,primary_sum,sample_count,coverage_complete,source_count,definition_revision)
          VALUES ('p','m','S2S','d','WER','v1','default','2026-10-01 11:00Z',5,1,true,1,2)
          ON CONFLICT DO NOTHING""")


def _publish(conn: Any) -> None:
    with conn.transaction():
        conn.execute("""UPDATE benchmarks_v2.dashboard_summary_state SET generation=7,
          as_of='2026-10-01 12:00Z',published_at='2026-10-01 12:00Z',definition_revision=2,
          definition_fingerprint='published' WHERE id=true""")
        for window in WINDOWS:
            conn.execute(f"REFRESH MATERIALIZED VIEW benchmarks_v2.normalized_results_{window}")


def _rows(conn: Any, table: str) -> list[Any]:
    return conn.execute(
        f"SELECT to_jsonb(t)-'metric_type' FROM benchmarks_v2.{table} t ORDER BY 1"
    ).fetchall()


def _payloads(conn: Any) -> dict[str, list[Any]]:
    tables = (
        *TABLES,
        "metrics",
        "dashboard_summary_state",
        "results",
        "results_by_bucket",
        *(f"normalized_results_{w}" for w in WINDOWS),
    )
    return {table: _rows(conn, table) for table in tables}


def _view_acl(conn: Any) -> list[Any]:
    return conn.execute("""SELECT c.relname,pg_get_userbyid(c.relowner),a.grantee,a.privilege_type,a.is_grantable
      FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
      CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) a
      WHERE n.nspname='benchmarks_v2' AND c.relkind='m' AND c.relname LIKE 'normalized_results_%'
      ORDER BY 1,2,3,4,5""").fetchall()


@pytest.fixture
def populated(compat: Any) -> Any:
    _seed(compat)
    _publish(compat)
    compat.execute(
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='metric_cleanup_reader') THEN CREATE ROLE metric_cleanup_reader NOLOGIN; END IF; END $$"
    )
    compat.execute("GRANT SELECT ON benchmarks_v2.normalized_results_24h TO PUBLIC")
    compat.execute(
        "GRANT SELECT ON benchmarks_v2.normalized_results_24h TO metric_cleanup_reader WITH GRANT OPTION"
    )
    compat.execute("GRANT USAGE ON SCHEMA benchmarks_v2 TO metric_cleanup_reader")
    compat.execute(
        "GRANT SELECT ON benchmarks_v2.dashboard_metric_values, benchmarks_v2.benchmark_observations, benchmarks_v2.runs, benchmarks_v2.dashboard_summary_state, benchmarks_v2.metrics TO metric_cleanup_reader"
    )
    compat.execute(
        "ALTER MATERIALIZED VIEW benchmarks_v2.normalized_results_7d OWNER TO metric_cleanup_reader"
    )
    return compat


def test_requires_explicit_opt_in(compat: Any) -> None:
    with pytest.raises(RuntimeError, match="opt-in"):
        _migrate(compat, CLEANUP)
    assert compat.execute("SELECT version_num FROM alembic_version").fetchone() == (COMPAT,)
    assert (
        compat.execute(
            "SELECT metric_type FROM benchmarks_v2.metric_evaluations LIMIT 0"
        ).description
        is not None
    )


@pytest.mark.parametrize(
    "damage", ["nullability", "foreign_key", "evaluation_index", "bucket_index"]
)
def test_preflight_refuses_incomplete_identity_invariants(compat: Any, damage: str) -> None:
    if damage == "nullability":
        compat.execute(
            "ALTER TABLE benchmarks_v2.metric_evaluations ALTER COLUMN metric_id DROP NOT NULL"
        )
    elif damage == "foreign_key":
        compat.execute(
            "ALTER TABLE benchmarks_v2.metric_evaluations DROP CONSTRAINT metric_evaluations_metric_id_fkey"
        )
    else:
        table = "metric_evaluations" if damage == "evaluation_index" else "metric_values_by_bucket"
        compat.execute(f"DROP INDEX benchmarks_v2.{table}_metric_identity_key")
        compat.execute(
            f"CREATE UNIQUE INDEX {table}_metric_identity_key ON benchmarks_v2.{table} (metric_id)"
        )
    with pytest.raises(RuntimeError, match="NOT NULL|foreign key|uniqueness index"):
        _migrate(compat, CLEANUP, opt_in=True)
    assert compat.execute("SELECT version_num FROM alembic_version").fetchone() == (COMPAT,)
    assert (
        compat.execute(
            "SELECT metric_type FROM benchmarks_v2.metric_evaluations LIMIT 0"
        ).description
        is not None
    )


def test_populated_cleanup_and_rollback_preserve_payloads_and_privileges(populated: Any) -> None:
    before = _payloads(populated)
    acl = _view_acl(populated)
    _migrate(populated, CLEANUP, opt_in=True)
    assert _payloads(populated) == before
    assert _view_acl(populated) == acl
    columns = populated.execute(
        """SELECT c.relname,a.attname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
      JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
      WHERE n.nspname='benchmarks_v2' AND (c.relname=ANY(%s) OR c.relname LIKE 'normalized_results_%%') AND a.attname='metric_type'""",
        (list(TABLES),),
    ).fetchall()
    assert columns == []
    assert populated.execute(
        "SELECT count(*) FROM pg_trigger WHERE tgname LIKE '%%sync_metric_identity' AND NOT tgisinternal"
    ).fetchone() == (0,)
    metrics = populated.execute("""SELECT m.code,v.avg_value FROM benchmarks_v2.normalized_results_24h v
      JOIN benchmarks_v2.metrics m ON m.id=v.metric_id WHERE v.dataset_id='d' ORDER BY m.code""").fetchall()
    assert dict(metrics) == pytest.approx(
        {"WER": 100 * 3 / 42, "TTFA": 150, "TTFARoundtrip": 35, "TTFALeadingSilence": 12.5}
    )
    _migrate(populated, COMPAT, down=True)
    assert _payloads(populated) == before
    assert _view_acl(populated) == acl
    assert populated.execute(
        "SELECT count(*) FROM pg_indexes WHERE schemaname='benchmarks_v2' AND indexname IN ('metric_evaluations_metric_identity_key','metric_values_by_bucket_metric_identity_key','dashboard_hourly_aggregates_metric_identity_key')"
    ).fetchone() == (3,)
    assert populated.execute(
        "SELECT count(*) FROM pg_trigger WHERE tgname LIKE '%%sync_metric_identity' AND NOT tgisinternal"
    ).fetchone() == (4,)
    # Both old code-only and new ID-only applications can write after rollback.
    observation = populated.execute(
        "SELECT observation_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()[0]
    for identity, value, variant in (("metric_type", "WER", "old"), ("metric_id", 1, "new")):
        row = populated.execute(
            f"""INSERT INTO benchmarks_v2.metric_evaluations
          (observation_id,{identity},metric_version,evaluation_variant,executor,status)
          VALUES (%s,%s,'v1',%s,'test','queued') RETURNING metric_type,metric_id""",
            (observation, value, variant),
        ).fetchone()
        assert row == ("WER", 1)
    _migrate(populated, CLEANUP, opt_in=True)
    assert _rows(populated, "normalized_results_24h") == before["normalized_results_24h"]


def test_unpopulated_views_remain_unpopulated(compat: Any) -> None:
    before = _rows(compat, "dashboard_summary_state")
    _migrate(compat, CLEANUP, opt_in=True)
    assert _rows(compat, "dashboard_summary_state") == before
    for window in WINDOWS:
        assert compat.execute(
            "SELECT relispopulated FROM pg_class WHERE oid=%s::regclass",
            (f"benchmarks_v2.normalized_results_{window}",),
        ).fetchone() == (False,)
    _migrate(compat, COMPAT, down=True)
    for window in WINDOWS:
        assert compat.execute(
            "SELECT relispopulated FROM pg_class WHERE oid=%s::regclass",
            (f"benchmarks_v2.normalized_results_{window}",),
        ).fetchone() == (False,)


def test_stale_published_payload_aborts_atomically(populated: Any) -> None:
    before = _payloads(populated)
    _seed(populated, sample="late")
    with pytest.raises(RuntimeError, match="parity failed.*refresh"):
        _migrate(populated, CLEANUP, opt_in=True)
    assert populated.execute("SELECT version_num FROM alembic_version").fetchone() == (COMPAT,)
    assert _rows(populated, "normalized_results_24h") == before["normalized_results_24h"]
    assert _rows(populated, "dashboard_summary_state") == before["dashboard_summary_state"]
    _publish(populated)
    _migrate(populated, CLEANUP, opt_in=True)
    assert populated.execute(
        "SELECT sum(sample_count) FROM benchmarks_v2.normalized_results_24h WHERE dataset_id='d'"
    ).fetchone() == (16,)


def test_lifecycle_catalog_and_projection_guards_survive(populated: Any) -> None:
    _migrate(populated, CLEANUP, opt_in=True)
    evaluation, observation, metric = populated.execute(
        "SELECT id,observation_id,metric_id FROM benchmarks_v2.metric_evaluations LIMIT 1"
    ).fetchone()
    with (
        pytest.raises(psycopg.errors.RaiseException, match="identity is immutable"),
        populated.transaction(),
    ):
        populated.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET metric_id=metric_id+1 WHERE id=%s",
            (evaluation,),
        )
    with (
        pytest.raises(psycopg.errors.RaiseException, match="terminal work rows"),
        populated.transaction(),
    ):
        populated.execute(
            "UPDATE benchmarks_v2.metric_evaluations SET status=status WHERE id=%s", (evaluation,)
        )
    with (
        pytest.raises(psycopg.errors.CheckViolation, match="disagrees with parent"),
        populated.transaction(),
    ):
        populated.execute(
            "UPDATE benchmarks_v2.dashboard_metric_values SET metric_id=metric_id+1 WHERE evaluation_id=%s",
            (evaluation,),
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="immutable"), populated.transaction():
        populated.execute("UPDATE benchmarks_v2.metrics SET code='changed' WHERE id=%s", (metric,))
    with (
        pytest.raises(psycopg.errors.RaiseException, match="terminal work payloads"),
        populated.transaction(),
    ):
        populated.execute(
            "UPDATE benchmarks_v2.metric_values SET value=value+1 WHERE metric_evaluation_id=%s",
            (evaluation,),
        )
    # The retained ID retry key rejects duplicate logical attempts.
    with pytest.raises(psycopg.errors.UniqueViolation), populated.transaction():
        populated.execute(
            """INSERT INTO benchmarks_v2.metric_evaluations (observation_id,metric_id,metric_version,evaluation_variant,executor,status)
          VALUES (%s,%s,'v1','default','test','queued')""",
            (observation, metric),
        )
    # Cascade deletion remains permitted when the observation itself is removed.
    populated.execute(
        "DELETE FROM benchmarks_v2.benchmark_observations WHERE id=%s", (observation,)
    )
    assert populated.execute(
        "SELECT count(*) FROM benchmarks_v2.metric_evaluations WHERE observation_id=%s",
        (observation,),
    ).fetchone() == (0,)
