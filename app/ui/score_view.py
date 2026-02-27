from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView


class ScoreView(QWebEngineView):
    """
    Engraved score renderer using OpenSheetMusicDisplay (OSMD) embedded in the app UI.

    This replaces the old MuseScore-PNG workflow:
      - We load a local HTML page (score.html) that bundles OSMD.
      - We inject MusicXML text directly into OSMD via JavaScript.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_ok: bool = False

        web_dir = Path(__file__).resolve().parent / "web"
        self._html_path = (web_dir / "score.html").resolve()

        self.loadFinished.connect(self._on_loaded)
        self.setUrl(QUrl.fromLocalFile(str(self._html_path)))

    def _on_loaded(self, ok: bool) -> None:
        self._loaded_ok = bool(ok)

    def is_ready(self) -> bool:
        return bool(self._loaded_ok)

    def render_musicxml_text(self, musicxml_text: str) -> None:
        """
        Render MusicXML (string) inside OSMD.

        Note: We pass XML as a JS template string to avoid file:// fetch + CORS issues.
        """
        if not self._loaded_ok:
            # If called too early (first load), re-load and let caller call again shortly.
            self.setUrl(QUrl.fromLocalFile(str(self._html_path)))
            return

        # Escape for JS template string
        s = musicxml_text.replace("\\", "\\\\")
        s = s.replace("`", "\\`")
        s = s.replace("${", "\\${")

        js = f"window.__renderMusicXML(`{s}`);"
        self.page().runJavaScript(js)