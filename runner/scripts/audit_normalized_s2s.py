#!/usr/bin/env python3
"""Run the normalized S2S parity audit against an explicitly supplied DSN."""

from __future__ import annotations

import argparse
import os
import sys

from coval_bench.migrations.audit_normalized_s2s import audit_s2s_dsn, report_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-stdin", action="store_true", help="Read a DSN privately from stdin")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument(
        "--explain", action="store_true", help="include a representative lookup plan"
    )
    args = parser.parse_args()
    dsn = sys.stdin.read(10000).strip() if args.dsn_stdin else os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("set DATABASE_URL or pipe the DSN with --dsn-stdin")
    report = audit_s2s_dsn(
        dsn,
        sample_limit=args.sample_limit,
        include_explain=args.explain,
    )
    print(report_json(report))
    return 0 if report.ready else 2


if __name__ == "__main__":
    sys.exit(main())
