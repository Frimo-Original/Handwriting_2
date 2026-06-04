from __future__ import annotations

import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import main_sync


class SyncTests(unittest.TestCase):
    def test_format_helpers(self) -> None:
        self.assertEqual(main_sync.format_bytes(512), "512 B")
        self.assertEqual(main_sync.format_bytes(1024), "1.0 KB")
        self.assertEqual(main_sync.format_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(main_sync.format_rate(2 * 1024 * 1024, 2.0), "1.00 MB/s")

    def test_safe_relative_path_rejects_traversal(self) -> None:
        self.assertEqual(main_sync.safe_relative_path("runs/gtx1660/best.pt"), Path("runs/gtx1660/best.pt"))
        with self.assertRaises(ValueError):
            main_sync.safe_relative_path("../secret.pt")
        with self.assertRaises(ValueError):
            main_sync.safe_relative_path("runs/../secret.pt")
        with self.assertRaises(ValueError):
            main_sync.safe_relative_path("/tmp/secret.pt")
        with self.assertRaises(ValueError):
            main_sync.safe_relative_path("C:/secret.pt")

    def test_selected_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epochs_dir = root / "runs"
            run = epochs_dir / "gtx1660" / "generator"
            run.mkdir(parents=True)
            for name in ["best.pt", "last.pt", "epoch_0005.pt", "config.json", "notes.txt"]:
                (run / name).write_text(name, encoding="utf-8")
            old_time = time.time() - 20
            for path in run.iterdir():
                path.touch()
                Path(path).touch()
                time.sleep(0.001)
                path.touch()
                path_time = old_time
                import os

                os.utime(path, (path_time, path_time))

            files = main_sync.selected_checkpoint_files(
                epochs_dir,
                include_epochs=True,
                include_best_last=True,
                include_configs=True,
                min_age=5,
            )
            names = {path.name for path in files}
            self.assertEqual(names, {"best.pt", "last.pt", "epoch_0005.pt", "config.json"})

            best_last = main_sync.selected_checkpoint_files(
                epochs_dir,
                include_epochs=False,
                include_best_last=True,
                include_configs=True,
                min_age=5,
            )
            self.assertEqual({path.name for path in best_last}, {"best.pt", "last.pt", "config.json"})

    def test_upload_file_to_local_server(self) -> None:
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            src = Path(src_tmp)
            dst = Path(dst_tmp)
            checkpoint = src / "gtx1660" / "generator" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            try:
                server = ThreadingHTTPServer(("127.0.0.1", 0), main_sync.SyncRequestHandler)
            except PermissionError as exc:
                self.skipTest(f"Local socket bind is not allowed in this sandbox: {exc}")
            server.root = dst  # type: ignore[attr-defined]
            server.token = "token"  # type: ignore[attr-defined]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                remote = f"http://127.0.0.1:{server.server_port}"
                result, elapsed = main_sync.upload_file(remote, src, checkpoint, token="token", timeout=5)
                self.assertEqual(result, "upload")
                self.assertGreaterEqual(elapsed, 0.0)
                self.assertEqual((dst / "gtx1660" / "generator" / "best.pt").read_bytes(), b"checkpoint")
                result, elapsed = main_sync.upload_file(remote, src, checkpoint, token="token", timeout=5)
                self.assertEqual(result, "skip")
                self.assertEqual(elapsed, 0.0)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
