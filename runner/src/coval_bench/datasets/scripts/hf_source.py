# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Turn a Hugging Face dataset path into builder ``Clip``s.

A source for the shared builder: it converts a dataset into ``Clip``s so the shared
build (clean → balance → transcode → manifest) is written once, never per dataset.
Other sources plug in the same way.

Reading:
    Uses the Hugging Face datasets-server REST API — rows come back as JSON with a
    downloadable audio URL — so there is no parquet reader / ``datasets`` dependency
    (``urllib`` + ``json`` only). Build-time module; the runtime loader never imports it.

Cost control (metadata-first):
    When the dataset has a duration column, only row *metadata* (transcript,
    duration, audio URL) is paged to run clean + balance; audio is downloaded for
    the *selected* clips alone — the whole dataset's audio is never pulled. With no
    duration column, it falls back to downloading all audio and reading each file's
    duration from its header.

Resolution outcomes (what can go wrong picking a dataset apart):
    HFNeedsChoice  — readable, but you must pick a config/split.
    HFAmbiguous    — readable, but you must name the audio/transcript column.
    HFUnsupported  — not readable at all (gated / not converted).
    The CLI turns the last two into a scaffolded handwritten adapter.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import soundfile as sf
import structlog

from coval_bench.datasets.scripts.framework import Clip

logger = structlog.get_logger(__name__)

_SERVER = "https://datasets-server.huggingface.co"
_PAGE = 100  # datasets-server hard cap on rows per /rows request
_STANDARD_SPLITS = frozenset({"train", "validation", "valid", "dev", "test"})
_TEXT_PRIORITY = ("transcript", "transcription", "text", "sentence", "normalized_text")
_DURATION_COLS = ("duration", "length", "duration_sec", "duration_seconds")
_UA = {"User-Agent": "coval-bench/build-dataset"}
_RETRIES = 6  # transient (429 / 5xx / timeout) attempts before giving up
_PAGE_PAUSE = 0.3  # seconds between pages, to stay under the rate limit
_SCAN_CAP_PER_SPLIT = 100  # cap rows scanned per split — full-split paging trips HF rate limits


class HFUnsupported(Exception):
    """datasets-server can't serve this dataset (gated / not converted / no rows)."""


class HFNetworkError(Exception):
    """A transient connection failure (timeout / reset / DNS) — safe to just retry."""


class HFNeedsChoice(Exception):
    """Readable, but the caller must pick a config/split (more than one available)."""


class HFAmbiguous(Exception):
    """Readable, but can't pin exactly one audio + one transcript column."""


def _api_get(endpoint: str, **params: object) -> Any:  # noqa: ANN401 (JSON payload is dynamic)
    url = f"{_SERVER}/{endpoint}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(_RETRIES):
        req = urllib.request.Request(url, headers=_UA)  # noqa: S310 (audited: hardcoded https HF URL)
        wait = 2 * (attempt + 1)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:  # 4xx (not rate-limit) = permanent
                raise HFUnsupported(f"{endpoint}: HTTP {exc.code}") from exc
            last = exc  # 429 rate-limit / 5xx = transient
            retry_after = exc.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
        except OSError as exc:  # URLError, timeouts, connection resets — transient
            last = exc
        if attempt < _RETRIES - 1:
            time.sleep(wait)
    # HTTP status exhausted → dataset-level (may be unconverted → parquet fallback);
    # a bare connection error → network problem the caller should just retry.
    if isinstance(last, urllib.error.HTTPError):
        raise HFUnsupported(f"datasets-server error for {endpoint}: HTTP {last.code}")
    raise HFNetworkError(f"datasets-server unreachable for {endpoint}: {last}")


def _fetch_file(src: str, dest: Path) -> None:
    """Stream *src* to *dest* atomically (chunked, so a huge file isn't held in RAM)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(src, headers=_UA)  # noqa: S310 (audited: HF cached-assets URL)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
            while chunk := resp.read(1 << 20):  # 1 MiB
                out.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)  # never leave a partial file behind
        raise HFNetworkError(f"download failed for {src}: {exc}") from exc
    tmp.replace(dest)


def _as_duration(value: object) -> float:
    """Coerce a duration cell to float; null/blank/non-numeric → 0.0 (then _clean drops it)."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def resolve_splits(hf_path: str, *, config: str | None, split: str | None) -> tuple[str, list[str]]:
    """Pick ``(config, [splits])`` for *hf_path*.

    Algorithm (no silent first-pick):
    - >1 config and none given → :class:`HFNeedsChoice` (caller must ``--config``).
    - standard partitions (train/validation/test) → use ``test`` (benchmarks are
      held-out); if no ``test``, the first standard split.
    - non-standard split names are *facets* (e.g. robustness conditions) → use ALL.
    - an explicit *split* always wins.
    """
    pairs = [
        (s["config"], s["split"]) for s in _api_get("splits", dataset=hf_path).get("splits", [])
    ]
    if not pairs:
        raise HFUnsupported(f"{hf_path}: datasets-server lists no splits")
    configs = sorted({c for c, _ in pairs})
    if config is None:
        if len(configs) > 1:
            raise HFNeedsChoice(f"{hf_path}: {len(configs)} configs {configs} — pass --config")
        config = configs[0]
    elif config not in configs:
        raise HFNeedsChoice(f"{hf_path}: config '{config}' not in {configs}")
    cfg_splits = [sp for c, sp in pairs if c == config]
    if split is not None:
        if split not in cfg_splits:
            raise HFNeedsChoice(f"{hf_path}: split '{split}' not in {cfg_splits}")
        return config, [split]
    standard = [sp for sp in cfg_splits if sp in _STANDARD_SPLITS]
    if standard:
        return config, ["test" if "test" in cfg_splits else standard[0]]
    return config, cfg_splits


def detect_columns(
    hf_path: str, config: str, split: str, *, audio_col: str | None, text_col: str | None
) -> tuple[str, str, str | None]:
    """Return ``(audio_col, transcript_col, duration_col | None)``, honoring overrides.

    Auto: audio = the single ``Audio``-typed column; transcript = the first string
    column in :data:`_TEXT_PRIORITY` (or the sole string column). Raises
    :class:`HFAmbiguous` when it can't pin exactly one of each.
    """
    feats = _api_get("rows", dataset=hf_path, config=config, split=split, offset=0, length=1).get(
        "features", []
    )
    names = {f["name"] for f in feats}
    audios = [f["name"] for f in feats if f["type"].get("_type") == "Audio"]
    strings = [f["name"] for f in feats if f["type"].get("dtype") == "string"]
    audio = audio_col or (audios[0] if len(audios) == 1 else None)
    text = text_col or next((n for n in _TEXT_PRIORITY if n in strings), None)
    if text is None and text_col is None and len(strings) == 1:
        text = strings[0]
    if audio is None or text is None or audio not in names or text not in names:
        raise HFAmbiguous(
            f"{hf_path}: audio={audios} strings={strings} — use --audio-col/--text-col"
        )
    duration = next((c for c in _DURATION_COLS if c in names), None)
    return audio, text, duration


def make_source(
    hf_path: str,
    config: str,
    splits: list[str],
    audio_col: str,
    text_col: str,
    duration_col: str | None,
) -> tuple[
    Callable[[Path], Path], Callable[[Path], list[Clip]], Callable[[list[Clip]], None] | None
]:
    """Build the ``(download, parse, fetch)`` framework hooks for this HF dataset.

    Metadata-first when *duration_col* is set (download indexes metadata only, fetch
    grabs audio for the selected clips); otherwise bulk (download grabs all audio,
    duration is read from each file, no fetch).
    """

    def _scan_split(
        split: str, audio_dir: Path, rows: list[dict[str, object]], *, with_audio: bool
    ) -> None:
        offset = 0
        got = 0
        while True:
            page = _api_get(
                "rows", dataset=hf_path, config=config, split=split, offset=offset, length=_PAGE
            )
            batch = page.get("rows", [])
            if not batch:
                break
            for entry in batch:
                got += 1
                row = entry["row"]
                cell = row[audio_col]
                src = (cell[0] if isinstance(cell, list) else cell)["src"]
                name = f"{split}-{entry['row_idx']}.wav"
                if with_audio and not (audio_dir / name).exists():
                    _fetch_file(src, audio_dir / name)
                meta = {
                    k: v
                    for k, v in row.items()
                    if k not in (audio_col, text_col, duration_col)
                    and isinstance(v, (str, int, float, bool))
                }
                meta["_split"] = split
                meta["_audio_src"] = src
                rows.append(
                    {
                        "audio": name,
                        "transcript": str(row[text_col] or ""),
                        "duration": _as_duration(row.get(duration_col)) if duration_col else 0.0,
                        "meta": meta,
                    }
                )
            offset += _PAGE
            total = int(page.get("num_rows_total", 0))
            if got >= _SCAN_CAP_PER_SPLIT and got < total:
                logger.info("hf_scan_capped", split=split, scanned=got, total=total)
                break
            if offset >= total:
                break
            time.sleep(_PAGE_PAUSE)  # stay polite under the rate limit

    def _index(cache_root: Path, *, with_audio: bool) -> Path:
        audio_dir = cache_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for split in splits:
            try:
                _scan_split(split, audio_dir, rows, with_audio=with_audio)
            except HFUnsupported as exc:  # one upstream-broken split shouldn't abort the build
                logger.warning("hf_split_skipped", split=split, error=str(exc))
        if not rows:
            raise HFUnsupported(f"{hf_path}: no rows loaded from any split")
        (cache_root / "index.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return cache_root

    def download(cache_root: Path) -> Path:  # index metadata (+ audio only in bulk mode)
        return _index(cache_root, with_audio=duration_col is None)

    def parse(source: Path) -> list[Clip]:  # index → clips
        audio_dir = source / "audio"
        clips: list[Clip] = []
        for line in (source / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record: Any = json.loads(line)
            path = audio_dir / str(record["audio"])
            # duration: use the dataset's duration column if it has one; otherwise
            # read it from the (bulk-downloaded) file header.
            if duration_col:
                duration = float(record["duration"])
            else:
                duration = float(sf.info(str(path)).duration)
            clips.append(
                Clip(
                    audio_path=path,
                    transcript=str(record["transcript"]),
                    duration_sec=duration,
                    meta=record["meta"],
                )
            )
        return clips

    def fetch(selected: list[Clip]) -> None:  # download audio for the selected clips only
        for clip in selected:
            if not clip.audio_path.exists():
                _fetch_file(str(clip.meta["_audio_src"]), clip.audio_path)

    return download, parse, (None if duration_col is None else fetch)


# --- parquet fallback: read the repo's parquet directly when the REST API can't ---
# Used when datasets-server is unavailable (unconverted dataset). Needs pyarrow
# (optional dep, build-time). Downloads the config's shard(s); metadata columns are
# read cheaply and audio bytes are extracted for the selected clips only.

_HF_API = "https://huggingface.co/api/datasets"
_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _config_parquet_files(files: list[str], config: str) -> list[str]:
    """Parquet paths for *config*: per-config dirs, else the single-config data/ layout."""
    parquet = sorted(f for f in files if f.endswith(".parquet"))
    matched = [f for f in parquet if f"/{config}/" in f]
    if not matched and config == "default":
        matched = [f for f in parquet if f.startswith("data/")]
    return matched


def _list_parquet_files(hf_path: str, config: str) -> list[str]:
    """Repo-relative parquet paths that belong to *config*."""
    req = urllib.request.Request(f"{_HF_API}/{hf_path}", headers=_UA)  # noqa: S310 (audited: HF API)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise HFUnsupported(f"{hf_path}: repo listing HTTP {exc.code}") from exc
    except OSError as exc:
        raise HFNetworkError(f"{hf_path}: repo listing unreachable: {exc}") from exc
    files = [str(s["rfilename"]) for s in data.get("siblings", [])]
    return _config_parquet_files(files, config)


def _detect_parquet_columns(
    schema: Any,  # noqa: ANN401 (pyarrow schema)
    audio_col: str | None,
    text_col: str | None,
) -> tuple[str, str, str | None]:
    """Pick (audio, transcript, duration|None) from a parquet schema; raise HFAmbiguous."""
    import pyarrow as pa

    names = list(schema.names)
    audios = [
        f.name for f in schema if pa.types.is_struct(f.type) and "bytes" in [c.name for c in f.type]
    ]
    strings = [f.name for f in schema if pa.types.is_string(f.type)]
    audio = audio_col or (audios[0] if len(audios) == 1 else None)
    text = text_col or next((n for n in _TEXT_PRIORITY if n in strings), None)
    if text is None and text_col is None and len(strings) == 1:
        text = strings[0]
    if audio is None or text is None:
        raise HFAmbiguous(f"parquet audio={audios} strings={strings} — use --audio-col/--text-col")
    duration = next((c for c in _DURATION_COLS if c in names), None)
    return audio, text, duration


def make_parquet_source(
    hf_path: str,
    config: str,
    audio_col: str | None,
    text_col: str | None,
) -> tuple[Callable[[Path], Path], Callable[[Path], list[Clip]], Callable[[list[Clip]], None]]:
    """Build (download, parse, fetch) hooks that read the repo's parquet directly.

    ``download`` pulls the config's shard(s); ``parse`` reads only metadata columns
    (transcript, duration, meta) — cheap — and requires a duration column so clean
    can run without decoding audio; ``fetch`` extracts audio bytes for the selected
    clips. Raises :class:`HFUnsupported` if pyarrow is missing or no shard exists.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise HFUnsupported("parquet fallback needs pyarrow (pip install pyarrow)") from exc
    files = _list_parquet_files(hf_path, config)
    if not files:
        raise HFUnsupported(f"{hf_path}: no parquet files for config '{config}'")
    detected: dict[str, str | None] = {}

    def _local(cache_root: Path, rel: str) -> Path:
        return cache_root / "parquet" / rel.replace("/", "__")

    def download(cache_root: Path) -> Path:
        (cache_root / "parquet").mkdir(parents=True, exist_ok=True)
        for rel in files:
            dest = _local(cache_root, rel)
            if not dest.exists():
                logger.info("hf_parquet_download", file=rel)
                _fetch_file(_HF_RESOLVE.format(repo=hf_path, path=rel), dest)
        return cache_root

    def parse(source: Path) -> list[Clip]:
        import pyarrow.parquet as pq

        audio_dir = source / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Clip] = []
        for rel in files:
            local = _local(source, rel)
            a, t, d = _detect_parquet_columns(pq.read_schema(local), audio_col, text_col)
            if d is None:
                raise HFUnsupported(
                    f"{hf_path}: parquet has no duration column — can't clean without audio"
                )
            detected.update(audio=a, text=t, duration=d)
            keep = [t, d, *(c for c in pq.read_schema(local).names if c not in (a, t, d))]
            for i, row in enumerate(pq.read_table(local, columns=keep).to_pylist()):
                meta: dict[str, object] = {
                    str(k): v
                    for k, v in row.items()
                    if k not in (t, d) and isinstance(v, (str, int, float, bool))
                }
                meta["_pq"] = str(local)
                meta["_row"] = i
                clips.append(
                    Clip(
                        audio_path=audio_dir / f"{local.stem}-{i}.wav",
                        transcript=str(row[t] or ""),
                        duration_sec=_as_duration(row[d]),
                        meta=meta,
                    )
                )
        return clips

    def fetch(selected: list[Clip]) -> None:
        import pyarrow.parquet as pq

        by_file: dict[str, list[Clip]] = {}
        for clip in selected:
            by_file.setdefault(str(clip.meta["_pq"]), []).append(clip)
        audio = detected["audio"]
        for pq_file, clips in by_file.items():
            # NOTE (deferred): this materializes the whole audio column per shard, so a
            # sparse fetch from a huge single-row-group file peaks at that file's size in
            # RAM. Acceptable for now — cache is deleted after the build and most shards
            # have many row groups. Proper fix = read only the selected rows' row groups.
            column = pq.read_table(pq_file, columns=[audio])[audio].to_pylist()
            for clip in clips:
                cell = column[int(clip.meta["_row"])]  # type: ignore[call-overload]
                data = cell["bytes"] if isinstance(cell, dict) else cell
                clip.audio_path.write_bytes(data)

    return download, parse, fetch


def scaffold_adapter(hf_path: str, dest: Path, *, detected: str) -> None:
    """Write a starter handwritten adapter (like SLURP) to complete by hand.

    The dataset id/slug is taken from *dest* (the file the CLI created) so the run
    instruction and ``SPEC.dataset_id`` match the file name exactly.
    """
    slug = dest.stem
    dest.write_text(
        f'''# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Handwritten adapter for {hf_path} — auto-ingest could not resolve it.

Reason: {detected}
Fill in download()/parse(), then run:  coval-build-dataset {slug}
"""

from __future__ import annotations

from pathlib import Path

from coval_bench.datasets.scripts.framework import Clip, DatasetSpec


def download(cache_root: Path) -> Path:
    raise NotImplementedError  # TODO: fetch the source into cache_root, return it


def parse(source: Path) -> list[Clip]:
    raise NotImplementedError  # TODO: source -> Clips (audio_path, transcript, duration_sec, meta)


SPEC = DatasetSpec(
    dataset_id="{slug}",
    cache_name="{slug}",
    download=download,
    parse=parse,
    dur_min=2.0,
    dur_max=10.0,
    min_words=3,
    num=50,
    dedup_key=lambda clip: clip.audio_path.name,
    balance_dims=(),  # TODO: add balance columns if wanted
    license="TODO",
    source="{hf_path}",
    needs_vad_offset=False,
)
''',
        encoding="utf-8",
    )
