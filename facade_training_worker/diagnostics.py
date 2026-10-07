from __future__ import annotations

import argparse
import json
from typing import Sequence

from .runtime import collect_runtime_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the locked Facade GPU Worker runtime.")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object.")
    parser.add_argument("--required-gpus", type=int, default=1)
    parser.add_argument("--pipeline", choices=(
        "initial_champion",
        "champion_failure_discovery",
        "challenger_update",
        "champion_challenger_evaluation",
    ))
    args = parser.parse_args(argv)
    if args.required_gpus < 1:
        parser.error("--required-gpus must be positive")
    report = collect_runtime_report(args.required_gpus, args.pipeline)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        state = "READY" if report["ready"] else "BLOCKED"
        print(f"Facade GPU Worker {report['worker_version']} ({report['phase']}): {state}")
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
