"""Desktop window for CleanSplit: a native window (Edge WebView2 on Windows) around the local server.

No Node, no Electron, no bundler: pywebview uses the WebView2 runtime that ships with Windows 10/11.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request


def _free_port(preferred: int = 8770) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def _wait_until_up(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"server did not start at {url}")


def launch(out_root: str = "outputs", port: int | None = None, dev: bool = False) -> None:
    import webview

    from .server import serve

    port = port or _free_port()
    url = f"http://127.0.0.1:{port}"
    threading.Thread(target=serve, kwargs={"port": port, "out_root": out_root}, daemon=True).start()
    _wait_until_up(url + "/api/state")
    window = webview.create_window(
        "CleanSplit", url, width=1380, height=860, min_size=(980, 620),
        background_color="#0B0C0D", text_select=False,
    )

    def on_files(paths):  # native drag-and-drop of audio files onto the window
        if paths:
            window.evaluate_js(f"window.cleansplitDropped({list(map(str, paths))!r})".replace("'", '"'))

    try:
        window.events.files_dropped += on_files
    except Exception:  # not supported on every backend
        pass
    webview.start(debug=dev)
