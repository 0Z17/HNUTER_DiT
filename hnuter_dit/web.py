"""Dependency-free local Web server for interactive HNUTER DiT inference."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .service import DEFAULT_COLLECTOR, DEFAULT_MODEL, DiTTrajectoryService


WEB_ROOT = Path(__file__).resolve().parent / "web_assets"
MAX_REQUEST_BYTES = 5_000_000


class DiTWebHandler(SimpleHTTPRequestHandler):
    service: DiTTrajectoryService

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("JSON 请求体必须在 1 byte 到 5 MB 之间")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON 请求体必须是对象")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/api/health", "/api/catalog"}:
            self._json(200, self.service.catalog())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "/api/scene/random": self.service.generate_scene_payload,
            "/api/endpoints/random": self.service.randomize_endpoints,
            "/api/validate": self.service.validate,
            "/api/sample": self.service.sample_trajectories,
        }
        operation = routes.get(urlsplit(self.path).path)
        if operation is None:
            self._json(404, {"error": "not found"})
            return
        try:
            self._json(200, operation(self._payload()))
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            self._json(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}")


def handler_type(service: DiTTrajectoryService) -> type[DiTWebHandler]:
    class BoundHandler(DiTWebHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    BoundHandler.service = service
    return BoundHandler


def serve(
    *, host: str = "127.0.0.1", port: int = 8790,
    checkpoint: Path = DEFAULT_MODEL, collector_root: Path = DEFAULT_COLLECTOR,
    device: str = "auto",
) -> None:
    service = DiTTrajectoryService(checkpoint, collector_root, device)
    server = ThreadingHTTPServer((host, port), handler_type(service))
    model = service.runtime.describe()
    print(
        f"HNUTER DiT Trajectory Lab: http://{host}:{port}\n"
        f"model={model['preset']} step={model['training_step']} "
        f"device={model['device']} sha256={model['checkpoint_sha256']}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--collector-root", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    serve(
        host=args.host, port=args.port, checkpoint=args.checkpoint,
        collector_root=args.collector_root, device=args.device,
    )


if __name__ == "__main__":
    main()
