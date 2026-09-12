"""Launch BioTransport Studio on loopback, without a frontend build step."""

import argparse

from .server import StudioServer


def main():
    parser = argparse.ArgumentParser(
        description="Open the local BioTransport workbench"
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    with StudioServer(args.port) as server:
        print(f"BioTransport Studio: http://127.0.0.1:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop. Experiments stay on this computer.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
