from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from typing import Sequence


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portable local CUDA process launcher without torch elastic.")
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.nproc != 1:
        parser.error("This release supports exactly one worker process (--nproc 1).")
    child_args = list(options.args)
    if child_args and child_args[0] == "--":
        child_args.pop(0)
    port = _free_local_port()
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for rank in range(options.nproc):
            environment = os.environ.copy()
            environment.update({
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                "WORLD_SIZE": str(options.nproc), "RANK": str(rank), "LOCAL_RANK": str(rank),
                "USE_LIBUV": "0",
            })
            processes.append(subprocess.Popen([sys.executable, "-m", options.module, *child_args], env=environment))
        while True:
            codes = [process.poll() for process in processes]
            failure = next((code for code in codes if code not in (None, 0)), None)
            if failure is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                return int(failure)
            if all(code == 0 for code in codes):
                return 0
            time.sleep(0.1)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
