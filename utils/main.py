"""
피노키오 프로젝트 (EyeCon) — 메인 윈도우
v3.1: 4분할 대시보드 레이아웃 + 대화 개선

┌──────────────────┬──────────────────────────┐
│   카메라 영상     │  감정 + 지표 패널         │
│  (랜드마크 표시)  │  영상 지표 / 음성 지표    │
├──────────────────┼──────────────────────────┤
│  실시간 그래프    │  대화 로그                │
│  (시계열 차트)    │  [피노키오] / [사용자]    │
└──────────────────┴──────────────────────────┘

실행: python main.py
"""
import sys
import os
import cv2
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QGridLayout, QFrame, QProgressBar,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

from config.settings import (
    CAMERA_INDEX, CAMERA_FPS, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    THRESHOLDS, UI,
)
from core.vision import VisionEngine
from core.voice_analyzer import VoiceAnalyzer
from core.llm_client import LLMClient
from utils.tts_engine import TTSEngine
from utils.stt_engine import STTWorker
from utils.logger import SessionLogger

# pyqtgraph 지연 로딩
try:
    import pyqtgraph as pg
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False
    print("[UI] pyqtgraph 미설치 — 그래프 비활성화 (pip install pyqtgraph)")


class IndicatorLabel(QWidget):
    """아이콘 + 수치 표시 위젯"""
    def __init__(self, icon: str, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.icon_label = QLabel(icon)
        self.icon_label.setFont(QFont("Segoe UI Emoji", 13))

        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        self.value_label = QLabel("—")
        self.value_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: bold;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)
        layout.addStretch()
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_value(self, text: str, color: str = "#ffffff"):
        self.value_label.setText(text)
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
        )


class PinocchioApp(QMainWindow):
    """피노키오 — 심리 상태 분석기 메인 윈도우"""

    # 스레드 안전 시그널 (백그라운드 → 메인 스레드)
    _tts_done_signal = pyqtSignal()
    _llm_response_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI["window_title"])
        self.setGeometry(*UI["window_geometry"])
        self.setStyleSheet("background-color: #0d0d0d;")

        # ── 시그널 연결 (메인 스레드에서 실행 보장) ──
        self._tts_done_signal.connect(self._on_tts_complete_safe)
        self._llm_response_signal.connect(self._on_llm_response_safe)

        # ── 코어 모듈 초기화 ──
        self.vision = VisionEngine()
        self.voice_analyzer = VoiceAnalyzer()
        self.llm = LLMClient()
        self.tts = TTSEngine()
        self.logger = SessionLogger()

        # ── 분석 상태 ──
        self.stress_score = 0
        self.blink_rate = 0.0
        self.iris_ratio_avg = 0.0
        self.gaze_direction = "center"

        # ── 음성 분석 결과 (최신) ──
        self._latest_voice_data = None
        self._tts_end_time = 0.0  # 응답 지연 계산용

        # ── 그래프 데이터 ──
        self._graph_time = []
        self._graph_stress = []
        self._graph_blink = []
        self._start_time = time.time()
        self._max_graph_points = 300

        # ── UI 구성 ──
        self._init_ui()

        # ── 카메라 시작 ──
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_frame)
        self.frame_timer.start(CAMERA_FPS)

        # ── 3초 후 대화 시작 ──
        QTimer.singleShot(3000, self._start_first_dialogue)

    # ══════════════════════════════════════════
    # UI 구성 — 4분할 대시보드
    # ══════════════════════════════════════════

    def _init_ui(self):
        central = QWidget()
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(10, 10, 10, 10)

        # ┌─────────┬─────────┐
        # │  카메라  │  지표   │
        # ├─────────┼─────────┤
        # │  그래프  │  대화   │
        # └─────────┴─────────┘
        grid.addWidget(self._create_camera_panel(), 0, 0)
        grid.addWidget(self._create_indicator_panel(), 0, 1)
        grid.addWidget(self._create_graph_panel(), 1, 0)
        grid.addWidget(self._create_chat_panel(), 1, 1)

        grid.setRowStretch(0, 6)
        grid.setRowStretch(1, 4)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 5)

        central.setLayout(grid)
        self.setCentralWidget(central)

    def _create_panel_frame(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background-color: #1a1a2e; "
            "border: 1px solid #2a2a4a; border-radius: 8px; }"
        )
        return frame

    # ── 좌상: 카메라 ──
    def _create_camera_panel(self) -> QWidget:
        frame = self._create_panel_frame()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("📷 실시간 영상")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 13px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        layout.addWidget(title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background: #000; border: 1px solid #333; border-radius: 4px;"
        )
        self.image_label.setMinimumSize(320, 240)
        layout.addWidget(self.image_label)

        frame.setLayout(layout)
        return frame

    # ── 우상: 지표 패널 ──
    def _create_indicator_panel(self) -> QWidget:
        frame = self._create_panel_frame()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # 감정 상태
        self.emotion_label = QLabel("😐 Neutral")
        self.emotion_label.setStyleSheet(
            "color: #ffffff; font-size: 20px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.emotion_label)

        # 종합 스코어 바
        score_row = QHBoxLayout()
        st = QLabel("종합 스코어")
        st.setStyleSheet("color: #aaa; font-size: 12px; border: none; background: transparent;")
        self.score_value_label = QLabel("0")
        self.score_value_label.setStyleSheet(
            "color: #00ff88; font-size: 16px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        score_row.addWidget(st)
        score_row.addStretch()
        score_row.addWidget(self.score_value_label)
        layout.addLayout(score_row)

        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedHeight(12)
        self.score_bar.setStyleSheet("""
            QProgressBar { background-color: #2a2a4a; border: none; border-radius: 6px; }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:0.5 #ffaa00, stop:1.0 #ff4444);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.score_bar)

        # ── 영상 지표 ──
        layout.addWidget(self._create_separator("영상 지표"))
        self.ind_gaze = IndicatorLabel("👁", "시선 불안정")
        self.ind_blink = IndicatorLabel("👀", "깜빡임")
        self.ind_pupil = IndicatorLabel("⚫", "동공 확장")
        self.ind_micro = IndicatorLabel("⚡", "미세 표정")
        self.ind_asym = IndicatorLabel("↔️", "비대칭")
        self.ind_lip = IndicatorLabel("👄", "입술 압축")
        for w in [self.ind_gaze, self.ind_blink, self.ind_pupil,
                  self.ind_micro, self.ind_asym, self.ind_lip]:
            layout.addWidget(w)

        # ── 음성 지표 ──
        layout.addWidget(self._create_separator("음성 지표"))
        self.ind_pitch = IndicatorLabel("🎵", "피치")
        self.ind_latency = IndicatorLabel("⏱️", "응답 지연")
        self.ind_disfluency = IndicatorLabel("💬", "비유창성")
        self.ind_speed = IndicatorLabel("📈", "말 속도")
        for w in [self.ind_pitch, self.ind_latency,
                  self.ind_disfluency, self.ind_speed]:
            layout.addWidget(w)

        layout.addStretch()
        frame.setLayout(layout)
        return frame

    def _create_separator(self, text: str) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background: transparent; border: none;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 6, 0, 2)

        line_l = QFrame()
        line_l.setFrameShape(QFrame.Shape.HLine)
        line_l.setStyleSheet("color: #3a3a5a; background: #3a3a5a; border: none;")
        line_l.setFixedHeight(1)

        label = QLabel(f" {text} ")
        label.setStyleSheet(
            "color: #6a6a9a; font-size: 11px; font-weight: bold; "
            "border: none; background: transparent;"
        )

        line_r = QFrame()
        line_r.setFrameShape(QFrame.Shape.HLine)
        line_r.setStyleSheet("color: #3a3a5a; background: #3a3a5a; border: none;")
        line_r.setFixedHeight(1)

        layout.addWidget(line_l, 1)
        layout.addWidget(label, 0)
        layout.addWidget(line_r, 1)
        widget.setLayout(layout)
        return widget

    # ── 좌하: 그래프 ──
    def _create_graph_panel(self) -> QWidget:
        frame = self._create_panel_frame()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("📊 실시간 그래프")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 13px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        layout.addWidget(title)

        if PG_AVAILABLE:
            try:
                pg.setConfigOptions(antialias=True)
                self.graph_widget = pg.PlotWidget()
                self.graph_widget.setBackground('#0d0d1a')
                self.graph_widget.showGrid(x=True, y=True, alpha=0.15)
                self.graph_widget.setYRange(0, 100)
                self.graph_widget.setMouseEnabled(x=False, y=False)

                # 축 라벨 스타일
                label_style = {'color': '#888', 'font-size': '11px'}
                self.graph_widget.setLabel('left', '값', **label_style)
                self.graph_widget.setLabel('bottom', '시간 (초)', **label_style)

                # 범례
                legend = self.graph_widget.addLegend(
                    offset=(10, 10), labelTextColor='#aaa'
                )

                # 그래프 라인
                self.stress_line = self.graph_widget.plot(
                    pen=pg.mkPen('#ff4444', width=2), name='스트레스'
                )
                self.blink_line = self.graph_widget.plot(
                    pen=pg.mkPen('#00d4ff', width=2), name='깜빡임/분'
                )

                layout.addWidget(self.graph_widget)
                print("[UI] pyqtgraph 그래프 초기화 완료")
            except Exception as e:
                print(f"[UI] pyqtgraph 초기화 실패: {e}")
                self._add_graph_placeholder(layout)
        else:
            self._add_graph_placeholder(layout)

        frame.setLayout(layout)
        return frame

    def _add_graph_placeholder(self, layout):
        """그래프 사용 불가 시 안내 메시지"""
        ph = QLabel(
            "그래프를 표시하려면 pyqtgraph를 설치하세요\n"
            "pip install pyqtgraph"
        )
        ph.setStyleSheet(
            "color: #666; font-size: 12px; border: none; background: transparent;"
        )
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ph)

    # ── 우하: 대화 로그 ──
    def _create_chat_panel(self) -> QWidget:
        frame = self._create_panel_frame()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("💬 대화 로그")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 13px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        layout.addWidget(title)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(
            "QTextEdit { background-color: #0d0d1a; color: #ffffff; "
            f"font-family: '{UI['font_family']}'; font-size: {UI['font_size']}px; "
            "border: 1px solid #2a2a4a; border-radius: 4px; padding: 8px; }"
        )
        layout.addWidget(self.log_display)

        self.status_label = QLabel("🟢 분석 대기")
        self.status_label.setStyleSheet(
            "color: #00ff88; font-size: 12px; border: none; background: transparent;"
        )
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    # ══════════════════════════════════════════
    # 프레임 업데이트
    # ══════════════════════════════════════════

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        face_data = self.vision.process_frame(frame)

        if face_data.detected:
            self.stress_score = min(100, int(self.vision.get_iris_instability()))
            self.blink_rate = self.vision.get_blink_rate()
            self.iris_ratio_avg = (
                face_data.iris_ratio_left + face_data.iris_ratio_right
            ) / 2.0
            self.gaze_direction = face_data.gaze_direction

            self._update_indicators(face_data)
            self._update_score_display()
            frame = self.vision.draw_landmarks(frame, face_data)

        # 그래프는 얼굴 감지 여부와 무관하게 항상 업데이트
        self._update_graph()

        # 화면 표시
        h, w, c = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        scaled = QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _update_score_display(self):
        self.score_bar.setValue(self.stress_score)
        self.score_value_label.setText(str(self.stress_score))

        if self.stress_score < THRESHOLDS["stress_low"]:
            c, s, t = "#00ff88", "🟢 분석 중", "#00ff88"
        elif self.stress_score < THRESHOLDS["stress_high"]:
            c, s, t = "#ffaa00", "🟡 긴장 감지", "#ffaa00"
        else:
            c, s, t = "#ff4444", "🔴 동요 감지", "#ff4444"

        self.score_value_label.setStyleSheet(
            f"color: {c}; font-size: 16px; font-weight: bold; "
            "border: none; background: transparent;"
        )
        self.status_label.setText(s)
        self.status_label.setStyleSheet(
            f"color: {t}; font-size: 12px; border: none; background: transparent;"
        )

    def _update_indicators(self, face_data):
        # ── 영상 지표 ──

        # 시선 불안정 (방향 + 회피 횟수)
        gaze_map = {"center": "정면", "left": "좌측", "right": "우측",
                    "up": "상단", "down": "하단"}
        aversion = face_data.gaze_aversion_count
        if aversion > 5:
            gc = "#ff4444"
            gaze_text = f"{gaze_map.get(self.gaze_direction, '—')} (회피 {aversion}회)"
        elif self.gaze_direction != "center":
            gc = "#ffaa00"
            gaze_text = f"{gaze_map.get(self.gaze_direction, '—')} ({face_data.gaze_fixation_sec:.0f}초)"
        else:
            gc = "#00ff88"
            gaze_text = "정면"
        self.ind_gaze.set_value(gaze_text, gc)

        # 깜빡임
        lo, hi = THRESHOLDS["blink_normal_range"]
        if self.blink_rate > hi:
            bc, bs = "#ff4444", " (↑)"
        elif 0 < self.blink_rate < lo:
            bc, bs = "#ffaa00", " (↓)"
        else:
            bc, bs = "#00ff88", ""
        self.ind_blink.set_value(f"{self.blink_rate:.0f}회/분{bs}", bc)

        # 동공 확장률
        dilation = face_data.pupil_dilation_pct
        if abs(dilation) > 8:
            pc = "#ff4444"
        elif abs(dilation) > 4:
            pc = "#ffaa00"
        else:
            pc = "#00ff88"
        sign = "+" if dilation >= 0 else ""
        self.ind_pupil.set_value(f"{sign}{dilation:.1f}%", pc)

        # 미세 표정
        mc = face_data.micro_expression_count
        if mc > 0:
            ml = face_data.micro_expression_label
            self.ind_micro.set_value(f"{mc}건 ({ml})", "#ff4444" if mc >= 3 else "#ffaa00")
        else:
            self.ind_micro.set_value("없음", "#00ff88")

        # 비대칭
        asym = face_data.asymmetry_score
        if asym > THRESHOLDS["asymmetry_threshold"]:
            ac = "#ff4444"
        elif asym > THRESHOLDS["asymmetry_threshold"] * 0.6:
            ac = "#ffaa00"
        else:
            ac = "#00ff88"
        self.ind_asym.set_value(f"{asym:.3f}", ac)

        # 입술 압축
        lip = face_data.lip_press_score
        if lip > THRESHOLDS["lip_press_threshold"]:
            lc = "#ff4444"
        elif lip > THRESHOLDS["lip_press_threshold"] * 0.6:
            lc = "#ffaa00"
        else:
            lc = "#00ff88"
        self.ind_lip.set_value(f"{lip:.2f}", lc)

        # ── 음성 지표 (최신 분석 결과 표시) ──
        vd = self._latest_voice_data
        if vd and vd.f0_mean > 0:
            # 피치
            self.ind_pitch.set_value(f"{vd.f0_mean:.0f}Hz (±{vd.f0_std:.0f})", "#ffffff")

            # 응답 지연
            lat = vd.response_latency_sec
            if lat > 3.0:
                lat_c = "#ff4444"
            elif lat > 2.0:
                lat_c = "#ffaa00"
            else:
                lat_c = "#00ff88"
            self.ind_latency.set_value(f"{lat:.1f}초", lat_c)

            # 비유창성
            dis = vd.disfluency_total
            if dis >= 3:
                dis_c = "#ff4444"
            elif dis >= 1:
                dis_c = "#ffaa00"
            else:
                dis_c = "#00ff88"
            self.ind_disfluency.set_value(f"{dis}회", dis_c)

            # 말 속도
            if vd.speech_rate > 0:
                self.ind_speed.set_value(f"{vd.speech_rate:.1f}음절/초", "#ffffff")
            else:
                self.ind_speed.set_value("—", "#666666")
        else:
            self.ind_pitch.set_value("대기 중", "#666666")
            self.ind_latency.set_value("대기 중", "#666666")
            self.ind_disfluency.set_value("대기 중", "#666666")
            self.ind_speed.set_value("대기 중", "#666666")

    def _update_graph(self):
        if not PG_AVAILABLE:
            return

        # 5프레임마다 업데이트 (부하 절감)
        if not hasattr(self, '_graph_frame_count'):
            self._graph_frame_count = 0
        self._graph_frame_count += 1
        if self._graph_frame_count % 5 != 0:
            return

        elapsed = time.time() - self._start_time
        self._graph_time.append(elapsed)
        self._graph_stress.append(self.stress_score)
        self._graph_blink.append(min(100, self.blink_rate))

        if len(self._graph_time) > self._max_graph_points:
            self._graph_time = self._graph_time[-self._max_graph_points:]
            self._graph_stress = self._graph_stress[-self._max_graph_points:]
            self._graph_blink = self._graph_blink[-self._max_graph_points:]

        self.stress_line.setData(self._graph_time, self._graph_stress)
        self.blink_line.setData(self._graph_time, self._graph_blink)

        # X축 자동 스크롤 (최근 60초 표시)
        if elapsed > 60:
            self.graph_widget.setXRange(elapsed - 60, elapsed)

    # ══════════════════════════════════════════
    # 대화 루프
    # ══════════════════════════════════════════

    def _start_first_dialogue(self):
        self._ai_turn(
            "안녕하세요~ 저는 피노키오예요! "
            "오늘 기분이 어떠세요? 편하게 얘기해 주세요~"
        )

    def _ai_turn(self, ai_text: str):
        color = UI["colors"]["ai_text"]
        self.log_display.append(
            f"<font color='{color}'><b>[피노키오]</b></font> {ai_text}"
        )
        self.logger.log_conversation("ai", ai_text)
        # TTS 완료 시 시그널 emit (메인 스레드에서 안전하게 처리)
        self.tts.speak(ai_text, on_complete=lambda: self._tts_done_signal.emit())

    def _on_tts_complete_safe(self):
        """TTS 완료 → 종료 시간 기록 + 0.5초 후 마이크 시작"""
        self._tts_end_time = time.time()
        QTimer.singleShot(500, self._launch_stt)

    def _launch_stt(self):
        color = UI["colors"]["system_text"]
        self.log_display.append(
            f"<font color='{color}'><i>(당신의 답변을 듣고 있습니다...)</i></font>"
        )
        self.stt_worker = STTWorker()
        self.stt_worker.result_signal.connect(self._handle_user_response)
        self.stt_worker.audio_signal.connect(self._handle_audio_file)
        self.stt_worker.timing_signal.connect(self._handle_stt_timing)
        self.stt_worker.start()

    def _handle_stt_timing(self, start_time: float):
        """발화 시작 시간 기록 → 응답 지연 계산용"""
        self._stt_start_time = start_time

    def _handle_audio_file(self, wav_path: str):
        """WAV 파일 경로 임시 저장 → 텍스트 도착 후 분석"""
        self._pending_wav_path = wav_path
        self._pending_latency = 0.0
        if hasattr(self, '_stt_start_time') and self._tts_end_time > 0:
            self._pending_latency = max(0, self._stt_start_time - self._tts_end_time)

    def _handle_user_response(self, text: str):
        color = UI["colors"]["user_text"]
        self.log_display.append(
            f"<font color='{color}'><b>[사용자]</b></font> {text}"
        )
        self.logger.log_conversation("user", text)

        # ── Phase 2B: 음성 분석 실행 (WAV + 텍스트 결합) ──
        if hasattr(self, '_pending_wav_path') and self._pending_wav_path:
            try:
                self._latest_voice_data = self.voice_analyzer.analyze(
                    wav_path=self._pending_wav_path,
                    text=text,
                    response_latency=getattr(self, '_pending_latency', 0.0),
                )
                self._pending_wav_path = None
                print(f"[Voice] 분석 완료: F0={self._latest_voice_data.f0_mean:.0f}Hz, "
                      f"지연={self._latest_voice_data.response_latency_sec:.1f}초")
            except Exception as e:
                print(f"[Voice] 분석 실패: {e}")

        prompt = self.llm.build_prompt(stress_score=self.stress_score, user_text=text)
        self.llm.generate_async(
            prompt, lambda ans: self._llm_response_signal.emit(ans or "")
        )

    def _on_llm_response_safe(self, answer: str):
        """LLM 응답 → 메인 스레드에서 다음 AI 턴"""
        if answer:
            self._ai_turn(answer)
        else:
            self._ai_turn("앗, 잠시 문제가 생겼어요. 다시 한번 말해줄 수 있어요?")

    # ══════════════════════════════════════════
    # 종료
    # ══════════════════════════════════════════

    def closeEvent(self, event):
        self.frame_timer.stop()
        self.cap.release()
        self.vision.release()
        self.tts.release()
        self.logger.save()
        print("[Pinocchio] 종료 완료")
        event.accept()


def main():
    app = QApplication(sys.argv)
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0d0d0d"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
    app.setPalette(palette)

    window = PinocchioApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
