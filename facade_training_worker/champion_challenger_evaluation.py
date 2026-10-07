from __future__ import annotations

from typing import Sequence

from .entrypoint import CONTRACTS, run


def main(argv: Sequence[str] | None = None) -> int:
    return run(CONTRACTS["champion_challenger_evaluation"], argv)


if __name__ == "__main__":
    raise SystemExit(main())
