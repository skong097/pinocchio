"""
피노키오 프로젝트 — 레이더 차트 위젯
13개 지표 (영상 6 + 음성 7)를 거미줄 차트로 시각화
QPainter 기반 커스텀 위젯
"""
import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


# 지표 라벨 (영상 6 + 음성 7 = 13개)
RADAR_LABELS = [
    # 영상
    "홍채\n불안정",
    "깜빡임\n변화",
    "동공\n확장",
    "미세\n표정",
    "비대칭",
    "입술\n압축",
    # 음성
    "피치\n변화",
    "응답\n지연",
    "비유창성",
    "Jitter",
    "말속도\n변화",
    "음량\n변화",
    # 종합
    "종합",
]

# 영상(파랑) / 음성(초록) / 종합(빨강)
RADAR_COLORS = (
    [QColor("#00d4ff")] * 6       # 영상: 시안
    + [QColor("#00ff88")] * 6     # 음성: 그린
    + [QColor("#ff4444")]         # 종합: 레드
)


class RadarChartWidget(QWidget):
    """13개 지표 레이더 차트"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._values = [0.0] * 13   # 0~100

    def set_values(self, values: list):
        """13개 값 업데이트 (0~100)"""
        if len(values) == 13:
            self._values = [min(100, max(0, v)) for v in values]
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 35  # 라벨 공간 확보

        n = len(self._values)
        angle_step = 2 * math.pi / n

        # ── 배경 ──
        painter.fillRect(self.rect(), QColor("#0d0d1a"))

        # ── 거미줄 (동심원 4단계) ──
        pen_grid = QPen(QColor(60, 60, 90, 80), 1)
        painter.setPen(pen_grid)
        for level in [0.25, 0.5, 0.75, 1.0]:
            r = radius * level
            points = []
            for i in range(n):
                angle = -math.pi / 2 + i * angle_step
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                points.append(QPointF(px, py))
            points.append(points[0])
            for j in range(len(points) - 1):
                painter.drawLine(points[j], points[j + 1])

        # ── 축선 ──
        pen_axis = QPen(QColor(60, 60, 90, 120), 1)
        painter.setPen(pen_axis)
        for i in range(n):
            angle = -math.pi / 2 + i * angle_step
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            painter.drawLine(QPointF(cx, cy), QPointF(px, py))

        # ── 데이터 영역 (채우기) ──
        data_points = []
        for i in range(n):
            angle = -math.pi / 2 + i * angle_step
            val = self._values[i] / 100.0
            r = radius * val
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            data_points.append(QPointF(px, py))

        # 채우기
        path = QPainterPath()
        path.moveTo(data_points[0])
        for pt in data_points[1:]:
            path.lineTo(pt)
        path.closeSubpath()

        fill_color = QColor(0, 212, 255, 40)
        painter.fillPath(path, QBrush(fill_color))

        # 외곽선
        pen_data = QPen(QColor(0, 212, 255, 180), 2)
        painter.setPen(pen_data)
        for i in range(n):
            painter.drawLine(data_points[i], data_points[(i + 1) % n])

        # ── 꼭짓점 (값에 따라 색상) ──
        for i in range(n):
            val = self._values[i]
            if val >= 70:
                dot_color = QColor("#ff4444")
            elif val >= 40:
                dot_color = QColor("#ffaa00")
            else:
                dot_color = QColor("#00ff88")

            painter.setPen(QPen(dot_color, 1))
            painter.setBrush(QBrush(dot_color))
            painter.drawEllipse(data_points[i], 4, 4)

        # ── 라벨 ──
        font = QFont("Sans", 8)
        painter.setFont(font)

        for i in range(n):
            angle = -math.pi / 2 + i * angle_step
            label_r = radius + 22
            lx = cx + label_r * math.cos(angle)
            ly = cy + label_r * math.sin(angle)

            painter.setPen(QPen(RADAR_COLORS[i], 1))

            # 텍스트 정렬
            text = RADAR_LABELS[i]
            lines = text.split("\n")
            line_h = 11
            total_h = len(lines) * line_h
            start_y = ly - total_h / 2

            for j, line in enumerate(lines):
                tw = painter.fontMetrics().horizontalAdvance(line)
                tx = lx - tw / 2
                ty = start_y + j * line_h + line_h
                painter.drawText(QPointF(tx, ty), line)

        painter.end()
