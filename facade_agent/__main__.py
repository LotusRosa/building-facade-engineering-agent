from __future__ import annotations

import argparse

from .server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Facade Model Maintenance Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        server = create_server("127.0.0.1", 0)
        server.server_close()
        print("FACADE_AGENT_CHECK_OK")
        return

    server = create_server(args.host, args.port)
    print(f"Facade Agent running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFacade Agent stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

