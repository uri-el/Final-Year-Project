from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt6.QtGui import QPen, QFont, QPainter
from PyQt6.QtCore import Qt


@dataclass
class NoteEvent:
    pitch: int
    onset_s: float
    dur_s: float
    string: int
    fret: int


class TabView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # time/grid
        self.dt = 0.10
        self.col_w = 22

        # layout
        self.line_gap = 28
        self.left_pad = 60
        self.top_pad = 45
        self.right_pad = 80

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # playhead (scene item)
        self._playhead_item: Optional[QGraphicsLineItem] = None
        self._playhead_s: float = 0.0

        # scene geometry cache
        self._cols: int = 0
        self._width: float = 0.0
        self._height: float = 0.0

        # sticky label style
        self._label_font = QFont("Consolas", 10)
        self._labels = ["e", "B", "G", "D", "A", "E"]  # top -> bottom

    def render_tab(self, notes: List[NoteEvent], bpm: float = 120.0):
        self.scene.clear()
        self._playhead_item = None

        if notes:
            last_t = max(float(n.onset_s) + float(n.dur_s) for n in notes)
        else:
            last_t = 5.0

        cols = int(last_t / self.dt) + 40
        width = self.left_pad + cols * self.col_w + self.right_pad
        height = self.top_pad + 5 * self.line_gap + 120

        self._cols = cols
        self._width = float(width)
        self._height = float(height)

        line_pen = QPen(Qt.GlobalColor.black)
        grid_pen = QPen(Qt.GlobalColor.darkGray)
        grid_pen.setStyle(Qt.PenStyle.DotLine)

        # string lines
        for i in range(6):
            y = self.top_pad + i * self.line_gap
            self.scene.addLine(self.left_pad, y, width - self.right_pad, y, line_pen)

        # beat grid
        beat = 60.0 / bpm if bpm and bpm > 0 else 0.5
        beat_cols = max(1, int(round(beat / self.dt)))
        for c in range(0, cols, beat_cols):
            x = self.left_pad + c * self.col_w
            self.scene.addLine(
                x, self.top_pad - 10,
                x, self.top_pad + 5 * self.line_gap + 10,
                grid_pen
            )

        # notes
        font = QFont("Consolas", 10)
        for n in notes:
            col = int(round(float(n.onset_s) / self.dt))
            x = self.left_pad + col * self.col_w
            s = max(1, min(6, int(n.string)))
            row = s - 1
            y = self.top_pad + row * self.line_gap

            t = self.scene.addText(str(int(n.fret)), font)
            t.setPos(x - 6, y - 12)

        # restore playhead after redraw
        self._draw_playhead(self._playhead_s)

        self.scene.setSceneRect(0, 0, width, height)
        self.viewport().update()

    def set_playhead(self, t_s: float):
        self._playhead_s = max(0.0, float(t_s))
        self._draw_playhead(self._playhead_s)

    def _draw_playhead(self, t_s: float):
        if self._width <= 0 or self._height <= 0:
            return

        col = int(round(float(t_s) / self.dt))
        col = max(0, min(self._cols, col))
        x = self.left_pad + col * self.col_w

        y0 = self.top_pad - 14
        y1 = self.top_pad + 5 * self.line_gap + 14

        if self._playhead_item is None:
            pen = QPen(Qt.GlobalColor.red)
            pen.setWidth(2)
            self._playhead_item = self.scene.addLine(x, y0, x, y1, pen)
            self._playhead_item.setZValue(10)
        else:
            self._playhead_item.setLine(x, y0, x, y1)

    # STICKY STRING LABELS: drawn in viewport coordinates, not scene coordinates
    def drawForeground(self, painter: QPainter, rect):
        super().drawForeground(painter, rect)

        painter.save()
        painter.setFont(self._label_font)
        painter.setPen(Qt.GlobalColor.white)

        # map each string y (scene) into view coordinates
        for i, lab in enumerate(self._labels):
            y_scene = self.top_pad + i * self.line_gap
            pt = self.mapFromScene(0, y_scene)
            painter.drawText(10, int(pt.y()) + 4, lab)

        painter.restore()