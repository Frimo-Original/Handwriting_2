from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_TOKEN = os.environ.get("HANDWRITING_SYNC_TOKEN", "handwriting-local-sync")
CHUNK_SIZE = 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EPOCHS_DIR = PROJECT_ROOT / "runs"


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0


def format_rate(bytes_count: int, seconds: float) -> str:
    if seconds <= 0:
        return "instant"
    return f"{bytes_count / 1024 / 1024 / seconds:.2f} MB/s"


def safe_relative_path(value: str) -> Path:
    raw = value.replace("\\", "/")
    if raw.startswith("/") or raw.startswith("~"):
        raise ValueError(f"Unsafe relative path: {value!r}")
    normalized = raw.strip("/")
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def selected_checkpoint_files(
    epochs_dir: Path,
    *,
    include_epochs: bool,
    include_best_last: bool,
    include_configs: bool,
    min_age: float,
) -> list[Path]:
    base = epochs_dir
    if not base.exists():
        return []

    patterns: list[str] = []
    if include_best_last:
        patterns.extend(["**/best.pt", "**/last.pt"])
    if include_epochs:
        patterns.append("**/epoch_*.pt")
    if include_configs:
        patterns.append("**/config.json")

    now = time.time()
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in base.glob(pattern):
            if not path.is_file():
                continue
            if min_age > 0 and now - path.stat().st_mtime < min_age:
                continue
            files[path] = None
    return sorted(files)


class SyncRequestHandler(BaseHTTPRequestHandler):
    server_version = "HandwritingSync/1.0"

    @property
    def root(self) -> Path:
        return self.server.root  # type: ignore[attr-defined]

    @property
    def token(self) -> str:
        return self.server.token  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.client_address[0]}] {format % args}")

    def _authorized(self) -> bool:
        return self.headers.get("X-Sync-Token") == self.token

    def _path_from_query(self) -> Path:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        return safe_relative_path(raw_path)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            json_response(self, 200, {"ok": True})
            return
        if parsed.path != "/stat":
            json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
            return
        if not self._authorized():
            json_response(self, 401, {"ok": False, "error": "bad token"})
            return
        try:
            relative = self._path_from_query()
        except ValueError as exc:
            json_response(self, 400, {"ok": False, "error": str(exc)})
            return

        target = self.root / relative
        if not target.exists() or not target.is_file():
            json_response(self, 404, {"ok": False, "exists": False})
            return
        stat = target.stat()
        json_response(
            self,
            200,
            {
                "ok": True,
                "exists": True,
                "size": stat.st_size,
                "sha256": file_sha256(target),
            },
        )

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/upload":
            json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
            return
        if not self._authorized():
            json_response(self, 401, {"ok": False, "error": "bad token"})
            return
        try:
            relative = self._path_from_query()
        except ValueError as exc:
            json_response(self, 400, {"ok": False, "error": str(exc)})
            return

        try:
            expected_size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            json_response(self, 400, {"ok": False, "error": "bad Content-Length"})
            return
        expected_hash = self.headers.get("X-File-Sha256")
        target = self.root / relative
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{time.time_ns()}")
        target.parent.mkdir(parents=True, exist_ok=True)

        received = 0
        digest = hashlib.sha256()
        started = time.perf_counter()
        try:
            with tmp.open("wb") as fh:
                remaining = expected_size
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    remaining -= len(chunk)

            actual_hash = digest.hexdigest()
            if received != expected_size:
                tmp.unlink(missing_ok=True)
                json_response(
                    self,
                    400,
                    {"ok": False, "error": f"incomplete upload: {received}/{expected_size}"},
                )
                return
            if expected_hash and actual_hash != expected_hash:
                tmp.unlink(missing_ok=True)
                json_response(
                    self,
                    400,
                    {"ok": False, "error": "sha256 mismatch", "sha256": actual_hash},
                )
                return
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        elapsed = time.perf_counter() - started
        print(
            f"RECV {relative.as_posix()} | {format_bytes(received)} | "
            f"{elapsed:.2f}s | {format_rate(received, elapsed)}",
            flush=True,
        )
        json_response(
            self,
            200,
            {"ok": True, "path": relative.as_posix(), "size": received, "sha256": actual_hash},
        )


def run_server(args: argparse.Namespace) -> None:
    epochs_dir = Path(args.epochs_dir).expanduser().resolve()
    epochs_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), SyncRequestHandler)
    server.root = epochs_dir  # type: ignore[attr-defined]
    server.token = args.token  # type: ignore[attr-defined]
    print(f"Receiving epochs into: {epochs_dir}")
    print(f"Listening on:          http://{args.host}:{args.port}")
    print("Stop with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping sync server.")
    finally:
        server.server_close()


def request_json(url: str, *, token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"X-Sync-Token": token})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def remote_matches(
    remote: str,
    relative: Path,
    *,
    token: str,
    size: int,
    sha256: str,
    timeout: float,
) -> bool:
    path = urllib.parse.quote(relative.as_posix())
    url = f"{remote.rstrip('/')}/stat?path={path}"
    try:
        payload = request_json(url, token=token, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return bool(payload.get("exists") and payload.get("size") == size and payload.get("sha256") == sha256)


def upload_file(
    remote: str,
    root: Path,
    path: Path,
    *,
    token: str,
    timeout: float,
) -> tuple[str, float]:
    relative = path.relative_to(root)
    digest = file_sha256(path)
    size = path.stat().st_size
    if remote_matches(remote, relative, token=token, size=size, sha256=digest, timeout=timeout):
        return "skip", 0.0

    path_query = urllib.parse.quote(relative.as_posix())
    url = f"{remote.rstrip('/')}/upload?path={path_query}"
    data = path.read_bytes()
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "X-Sync-Token": token,
            "X-File-Sha256": digest,
            "Content-Type": "application/octet-stream",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    if not payload.get("ok"):
        raise RuntimeError(f"Upload failed for {relative}: {payload}")
    return "upload", elapsed


def run_push_once(args: argparse.Namespace) -> tuple[int, int]:
    epochs_dir = Path(args.epochs_dir).expanduser().resolve()
    files = selected_checkpoint_files(
        epochs_dir,
        include_epochs=not args.best_last_only,
        include_best_last=True,
        include_configs=args.include_configs,
        min_age=args.min_age,
    )
    uploaded = 0
    skipped = 0
    uploaded_bytes = 0
    started = time.perf_counter()
    print(
        f"Scan {time.strftime('%H:%M:%S')}: {epochs_dir} | files={len(files)} | "
        f"mode={'best/last only' if args.best_last_only else 'all epochs'}",
        flush=True,
    )
    for path in files:
        relative = path.relative_to(epochs_dir)
        size = path.stat().st_size
        print(f"FILE {relative} | {format_bytes(size)}", flush=True)
        try:
            result, elapsed = upload_file(args.remote, epochs_dir, path, token=args.token, timeout=args.timeout)
        except Exception as exc:
            print(f"ERROR {relative}: {exc}", file=sys.stderr, flush=True)
            continue
        if result == "upload":
            uploaded += 1
            uploaded_bytes += size
            print(
                f"UP   {relative} | {format_bytes(size)} | "
                f"{elapsed:.2f}s | {format_rate(size, elapsed)}",
                flush=True,
            )
        else:
            skipped += 1
            print(f"SKIP {relative} | {format_bytes(size)} | already on receiver", flush=True)
    total_elapsed = time.perf_counter() - started
    print(
        f"Done. uploaded={uploaded}, skipped={skipped}, found={len(files)}, "
        f"sent={format_bytes(uploaded_bytes)}, elapsed={total_elapsed:.2f}s, "
        f"avg={format_rate(uploaded_bytes, total_elapsed)}",
        flush=True,
    )
    return uploaded, skipped


def run_push(args: argparse.Namespace) -> None:
    run_push_once(args)


def run_watch(args: argparse.Namespace) -> None:
    print(f"Watching epochs: {Path(args.epochs_dir).expanduser().resolve()}")
    print(f"Remote: {args.remote}")
    print("Stop with Ctrl+C.")
    try:
        while True:
            run_push_once(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping sync watcher.")


def add_sender_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", required=True, help="Receiver URL, for example http://192.168.1.20:8765")
    parser.add_argument(
        "--epochs-dir",
        default=str(DEFAULT_EPOCHS_DIR),
        help="Directory with training checkpoints. Defaults to this project's runs/ folder.",
    )
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--min-age",
        type=float,
        default=5.0,
        help="Skip files modified less than this many seconds ago.",
    )
    parser.add_argument(
        "--best-last-only",
        action="store_true",
        help="Send only best.pt and last.pt; skip epoch_*.pt.",
    )
    parser.add_argument(
        "--include-configs",
        action="store_true",
        help="Also send config.json files from the epochs directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer training epoch/checkpoint files between local-network PCs.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Receive checkpoint files into this project's runs/ folder.")
    serve.add_argument(
        "--epochs-dir",
        default=str(DEFAULT_EPOCHS_DIR),
        help="Directory to receive training checkpoints into. Defaults to this project's runs/ folder.",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token", default=DEFAULT_TOKEN)
    serve.set_defaults(func=run_server)

    push = sub.add_parser("push", help="Send checkpoint files once from this project's runs/ folder.")
    add_sender_args(push)
    push.set_defaults(func=run_push)

    watch = sub.add_parser("watch", help="Continuously send new/changed checkpoint files from runs/.")
    add_sender_args(watch)
    watch.add_argument("--interval", type=float, default=60.0, help="Scan interval in seconds.")
    watch.set_defaults(func=run_watch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
