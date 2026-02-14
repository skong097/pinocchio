"""
피노키오 프로젝트 — Vision 모듈 v3.2 (Phase 2A)
MediaPipe FaceLandmarker (478점 + 52 Blendshapes) 기반

Phase 2A 추가:
  - 미세 표정 감지 (Blendshapes 급변 감지)
  - 동공 확장률 (베이스라인 대비 %)
  - 시선 고정/회피 패턴 (시간 기반)
  - 표정 비대칭 종합 (다중 AU)
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from config.settings import (
    FACE_LANDMARKER, FACE_LANDMARKER_MODEL,
    IRIS_LANDMARKS, EYE_LANDMARKS,
    THRESHOLDS, BLENDSHAPE_AU_MAP,
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[Vision] MediaPipe를 설치해주세요: pip install mediapipe")


@dataclass
class FaceData:
    """한 프레임의 얼굴 분석 결과"""
    # 랜드마크
    landmarks: Optional[list] = None
    left_iris: Optional[tuple] = None
    right_iris: Optional[tuple] = None
    blendshapes: Optional[dict] = None

    # 눈 깜빡임
    ear_left: float = 0.0
    ear_right: float = 0.0
    is_blinking: bool = False

    # 동공 크기
    iris_ratio_left: float = 0.0
    iris_ratio_right: float = 0.0
    pupil_dilation_pct: float = 0.0    # 베이스라인 대비 변화율 (%)

    # 시선
    gaze_direction: str = "center"
    gaze_fixation_sec: float = 0.0     # 현재 방향 고정 시간 (초)
    gaze_aversion_count: int = 0       # 최근 1분 시선 회피 횟수

    # 미세 표정
    micro_expression_count: int = 0    # 최근 감지된 미세 표정 수
    micro_expression_label: str = ""   # 마지막 감지된 미세 표정 유형

    # 비대칭
    asymmetry_score: float = 0.0       # 종합 비대칭 점수

    # 입술 압축
    lip_press_score: float = 0.0

    # 메타
    timestamp: float = 0.0
    detected: bool = False


class VisionEngine:
    """
    MediaPipe FaceLandmarker 기반 실시간 얼굴 분석 엔진 v3.2
    Phase 2A: 13개 영상 지표 중 6개 실시간 추출
    """

    def __init__(self):
        self.landmarker = None
        self._init_landmarker()

        # ── 깜빡임 ──
        self._blink_counter = 0
        self._blink_total = 0
        self._blink_timestamps = deque(maxlen=600)

        # ── 홍채 ──
        self._iris_history = deque(maxlen=THRESHOLDS["iris_history_size"])

        # ── 동공 베이스라인 ──
        self._pupil_baseline_samples = deque(maxlen=300)  # 최초 10초 (~300프레임)
        self._pupil_baseline = None  # 평균 iris_ratio (베이스라인 확정 후)

        # ── 시선 패턴 ──
        self._gaze_current_dir = "center"
        self._gaze_dir_start_time = time.time()
        self._gaze_aversion_timestamps = deque(maxlen=300)

        # ── 미세 표정 (Blendshapes 급변 감지) ──
        self._prev_blendshapes = {}
        self._prev_bs_time = 0.0
        self._micro_expr_timestamps = deque(maxlen=100)
        self._micro_expr_last_label = ""

        # 미세 표정 감지 대상 Blendshapes
        self._micro_targets = [
            "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight",
            "browDownLeft", "browDownRight",
            "browInnerUp",
            "eyeSquintLeft", "eyeSquintRight",
            "jawOpen",
            "mouthPressLeft", "mouthPressRight",
        ]

        # 미세 표정 → 라벨 매핑
        self._micro_label_map = {
            "mouthSmileLeft": "억제된 웃음",
            "mouthSmileRight": "억제된 웃음",
            "mouthFrownLeft": "불쾌 억제",
            "mouthFrownRight": "불쾌 억제",
            "browDownLeft": "미간 찌푸림",
            "browDownRight": "미간 찌푸림",
            "browInnerUp": "걱정/놀람",
            "eyeSquintLeft": "의심/불편",
            "eyeSquintRight": "의심/불편",
            "jawOpen": "놀람",
            "mouthPressLeft": "입술 압축",
            "mouthPressRight": "입술 압축",
        }

        # ── 비대칭 쌍 ──
        self._asymmetry_pairs = [
            ("mouthSmileLeft", "mouthSmileRight"),
            ("mouthFrownLeft", "mouthFrownRight"),
            ("browDownLeft", "browDownRight"),
            ("browOuterUpLeft", "browOuterUpRight"),
            ("eyeSquintLeft", "eyeSquintRight"),
            ("cheekSquintLeft", "cheekSquintRight"),
        ]

    def _init_landmarker(self):
        if not MP_AVAILABLE:
            return
        model_path = str(FACE_LANDMARKER_MODEL)
        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=FACE_LANDMARKER["max_num_faces"],
                min_face_detection_confidence=FACE_LANDMARKER["min_detection_confidence"],
                min_face_presence_confidence=FACE_LANDMARKER["min_tracking_confidence"],
                min_tracking_confidence=FACE_LANDMARKER["min_tracking_confidence"],
                output_face_blendshapes=FACE_LANDMARKER["output_blendshapes"],
                output_facial_transformation_matrixes=FACE_LANDMARKER["output_face_transformation_matrixes"],
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[Vision] FaceLandmarker 초기화 완료 (478점 + 52 Blendshapes)")
        except Exception as e:
            print(f"[Vision] FaceLandmarker 초기화 실패: {e}")
            self.landmarker = None

    # ══════════════════════════════════════════
    # 메인 프레임 처리
    # ══════════════════════════════════════════

    def process_frame(self, frame: np.ndarray) -> FaceData:
        face_data = FaceData(timestamp=time.time())

        if self.landmarker is None:
            return face_data

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = self.landmarker.detect(mp_image)
        except Exception as e:
            print(f"[Vision] 감지 실패: {e}")
            return face_data

        if not result.face_landmarks:
            return face_data

        landmarks = result.face_landmarks[0]
        face_data.detected = True
        face_data.landmarks = landmarks

        # ── 홍채 좌표 ──
        li = IRIS_LANDMARKS["left_center"]
        ri = IRIS_LANDMARKS["right_center"]
        face_data.left_iris = (landmarks[li].x, landmarks[li].y)
        face_data.right_iris = (landmarks[ri].x, landmarks[ri].y)
        self._iris_history.append([
            landmarks[li].x, landmarks[li].y,
            landmarks[ri].x, landmarks[ri].y
        ])

        # ── Blendshapes ──
        if result.face_blendshapes:
            bs = result.face_blendshapes[0]
            face_data.blendshapes = {cat.category_name: cat.score for cat in bs}

        # ── EAR + 깜빡임 ──
        face_data.ear_left = self._compute_ear(landmarks, EYE_LANDMARKS["left"])
        face_data.ear_right = self._compute_ear(landmarks, EYE_LANDMARKS["right"])
        avg_ear = (face_data.ear_left + face_data.ear_right) / 2.0
        if avg_ear < THRESHOLDS["ear_blink"]:
            self._blink_counter += 1
        else:
            if self._blink_counter >= THRESHOLDS["blink_consec_frames"]:
                self._blink_total += 1
                self._blink_timestamps.append(time.time())
                face_data.is_blinking = True
            self._blink_counter = 0

        # ── 동공 크기 + 확장률 ──
        face_data.iris_ratio_left = self._compute_iris_ratio(
            landmarks, EYE_LANDMARKS["left"], IRIS_LANDMARKS["left_ring"]
        )
        face_data.iris_ratio_right = self._compute_iris_ratio(
            landmarks, EYE_LANDMARKS["right"], IRIS_LANDMARKS["right_ring"]
        )
        avg_ratio = (face_data.iris_ratio_left + face_data.iris_ratio_right) / 2.0
        face_data.pupil_dilation_pct = self._compute_pupil_dilation(avg_ratio)

        # ── 시선 방향 + 고정/회피 ──
        face_data.gaze_direction = self._estimate_gaze(landmarks)
        self._update_gaze_pattern(face_data)

        # ── 미세 표정 감지 ──
        if face_data.blendshapes:
            self._detect_micro_expression(face_data)

        # ── 비대칭 종합 ──
        if face_data.blendshapes:
            face_data.asymmetry_score = self._compute_asymmetry(face_data.blendshapes)

        # ── 입술 압축 ──
        if face_data.blendshapes:
            lp_l = face_data.blendshapes.get("mouthPressLeft", 0)
            lp_r = face_data.blendshapes.get("mouthPressRight", 0)
            face_data.lip_press_score = (lp_l + lp_r) / 2.0

        return face_data

    # ══════════════════════════════════════════
    # Phase 2A: 동공 확장률
    # ══════════════════════════════════════════

    def _compute_pupil_dilation(self, current_ratio: float) -> float:
        """
        동공 확장률 (베이스라인 대비 %)
        최초 ~10초(300프레임) 자동 베이스라인 수집 후 비교
        """
        if self._pupil_baseline is None:
            # 베이스라인 수집 중
            self._pupil_baseline_samples.append(current_ratio)
            if len(self._pupil_baseline_samples) >= 300:
                self._pupil_baseline = float(np.mean(self._pupil_baseline_samples))
                print(f"[Vision] 동공 베이스라인 확정: {self._pupil_baseline:.4f}")
            return 0.0

        if self._pupil_baseline <= 0:
            return 0.0

        change_pct = ((current_ratio - self._pupil_baseline)
                      / self._pupil_baseline * 100)
        return round(change_pct, 1)

    # ══════════════════════════════════════════
    # Phase 2A: 시선 고정/회피 패턴
    # ══════════════════════════════════════════

    def _update_gaze_pattern(self, face_data: FaceData):
        """시선 방향 변화 추적 → 고정 시간, 회피 횟수"""
        now = time.time()
        new_dir = face_data.gaze_direction

        if new_dir != self._gaze_current_dir:
            # 방향이 바뀜 = 시선 이동
            if self._gaze_current_dir == "center" and new_dir != "center":
                # 정면 → 다른 곳 = 회피
                self._gaze_aversion_timestamps.append(now)
            self._gaze_current_dir = new_dir
            self._gaze_dir_start_time = now

        # 현재 방향 고정 시간
        face_data.gaze_fixation_sec = round(now - self._gaze_dir_start_time, 1)

        # 최근 60초 회피 횟수
        cutoff = now - 60.0
        face_data.gaze_aversion_count = sum(
            1 for t in self._gaze_aversion_timestamps if t > cutoff
        )

    # ══════════════════════════════════════════
    # Phase 2A: 미세 표정 감지
    # ══════════════════════════════════════════

    def _detect_micro_expression(self, face_data: FaceData):
        """
        Blendshapes 급변 감지 → 미세 표정
        0.5초 이내에 delta > threshold 변화가 있으면 미세 표정으로 판정
        """
        now = face_data.timestamp
        bs = face_data.blendshapes

        if self._prev_blendshapes and (now - self._prev_bs_time) < THRESHOLDS["micro_expr_duration"]:
            for key in self._micro_targets:
                curr = bs.get(key, 0)
                prev = self._prev_blendshapes.get(key, 0)
                delta = abs(curr - prev)

                if delta > THRESHOLDS["micro_expr_delta"]:
                    self._micro_expr_timestamps.append(now)
                    self._micro_expr_last_label = self._micro_label_map.get(key, key)
                    break  # 한 프레임에 한 건만

        # 최근 60초 미세 표정 수
        cutoff = now - 60.0
        face_data.micro_expression_count = sum(
            1 for t in self._micro_expr_timestamps if t > cutoff
        )
        face_data.micro_expression_label = self._micro_expr_last_label

        # 현재 프레임 저장
        self._prev_blendshapes = dict(bs)
        self._prev_bs_time = now

    # ══════════════════════════════════════════
    # Phase 2A: 비대칭 종합
    # ══════════════════════════════════════════

    def _compute_asymmetry(self, blendshapes: dict) -> float:
        """다중 좌/우 AU 쌍의 평균 비대칭"""
        diffs = []
        for left_key, right_key in self._asymmetry_pairs:
            l_val = blendshapes.get(left_key, 0)
            r_val = blendshapes.get(right_key, 0)
            # 둘 다 거의 0이면 의미 없으므로 제외
            if l_val > 0.05 or r_val > 0.05:
                diffs.append(abs(l_val - r_val))
        if not diffs:
            return 0.0
        return round(float(np.mean(diffs)), 3)

    # ══════════════════════════════════════════
    # 기존 메서드 (유지)
    # ══════════════════════════════════════════

    def get_blink_rate(self, window_sec: float = 60.0) -> float:
        now = time.time()
        recent = [t for t in self._blink_timestamps if now - t <= window_sec]
        if window_sec <= 0:
            return 0.0
        return len(recent) * (60.0 / window_sec)

    def get_iris_instability(self) -> float:
        if len(self._iris_history) < THRESHOLDS["iris_history_size"]:
            return 0.0
        return float(np.std(list(self._iris_history), axis=0).mean() * 3000)

    def draw_landmarks(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """썬글라스 오버레이 + 홍채 추적 표시"""
        if not face_data.detected or face_data.landmarks is None:
            return frame

        h, w = frame.shape[:2]
        annotated = frame.copy()
        lm = face_data.landmarks

        # ── 눈 사이 거리 기반 고정 렌즈 크기 계산 ──
        if not face_data.left_iris or not face_data.right_iris:
            return frame

        l_iris_x = face_data.left_iris[0] * w
        r_iris_x = face_data.right_iris[0] * w
        eye_dist = abs(r_iris_x - l_iris_x)  # 양쪽 홍채 사이 거리 (픽셀)

        fixed_lens_w = int(eye_dist * 0.29)   # 고정 가로 (0.22 * 1.3)
        fixed_lens_h = int(eye_dist * 0.18)   # 고정 세로

        # ── 양쪽 눈 렌즈 그리기 ──
        eye_data = []

        for eye_key, iris_key in [("left", "left_iris"), ("right", "right_iris")]:
            eye_indices = EYE_LANDMARKS[eye_key]
            iris = getattr(face_data, iris_key)
            if not iris:
                continue

            eye_pts = [(lm[i].x * w, lm[i].y * h) for i in eye_indices]
            xs = [p[0] for p in eye_pts]

            # 렌즈 중심 = 홍채 위치
            cx = int(iris[0] * w)
            cy = int(iris[1] * h)

            lens_w = fixed_lens_w
            lens_h = fixed_lens_h

            eye_data.append((cx, cy, lens_w, lens_h, xs))

            # ── 반투명 노란색 렌즈 (위가 넓은 사다리꼴) ──
            top_left = (cx - int(lens_w * 1.2), cy - lens_h)
            top_right = (cx + int(lens_w * 1.2), cy - lens_h)
            bottom_right = (cx + int(lens_w * 0.7), cy + lens_h)
            bottom_left = (cx - int(lens_w * 0.7), cy + lens_h)

            trapezoid = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

            overlay = annotated.copy()
            cv2.fillPoly(overlay, [trapezoid], (0, 200, 255))  # BGR: 노란색
            cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

            # ── 렌즈 테두리 ──
            cv2.polylines(annotated, [trapezoid], True, (40, 40, 40), 2)

            # ── 반사광 하이라이트 ──
            hl_x = cx - int(lens_w * 0.4)
            hl_y = cy - int(lens_h * 0.4)
            cv2.ellipse(annotated, (hl_x, hl_y),
                        (int(lens_w * 0.2), int(lens_h * 0.15)),
                        -30, 0, 360, (255, 255, 255), 1)

            # ── 홍채 위치 (틴트 위 초록 점) ──
            iris_x, iris_y = int(iris[0] * w), int(iris[1] * h)
            cv2.circle(annotated, (iris_x, iris_y), 3, (0, 255, 0), -1)

        # ── 브릿지 + 템플 (양쪽 눈 데이터 필요) ──
        if len(eye_data) == 2 and face_data.left_iris and face_data.right_iris:
            l_cx, l_cy, l_lw, l_lh, l_xs = eye_data[0]
            r_cx, r_cy, r_lw, r_lh, r_xs = eye_data[1]

            # 브릿지: 왼쪽 렌즈 안쪽 → 코 → 오른쪽 렌즈 안쪽
            nose_pt = (int(lm[6].x * w), int(lm[6].y * h))
            l_inner = (l_cx + l_lw, l_cy)
            r_inner = (r_cx - r_lw, r_cy)
            bridge_mid = (nose_pt[0], min(l_cy, r_cy) - 5)
            cv2.line(annotated, l_inner, bridge_mid, (40, 40, 40), 2)
            cv2.line(annotated, bridge_mid, r_inner, (40, 40, 40), 2)

            # 템플 (다리) — 관자놀이까지
            l_temple_start = (l_cx - l_lw, l_cy)
            l_temple_end = (int(lm[234].x * w), int(lm[234].y * h))
            cv2.line(annotated, l_temple_start, l_temple_end, (40, 40, 40), 2)

            r_temple_start = (r_cx + r_lw, r_cy)
            r_temple_end = (int(lm[454].x * w), int(lm[454].y * h))
            cv2.line(annotated, r_temple_start, r_temple_end, (40, 40, 40), 2)

        return annotated

    @staticmethod
    def _compute_ear(landmarks, eye_indices) -> float:
        p = [landmarks[i] for i in eye_indices]
        v1 = np.sqrt((p[1].x - p[5].x) ** 2 + (p[1].y - p[5].y) ** 2)
        v2 = np.sqrt((p[2].x - p[4].x) ** 2 + (p[2].y - p[4].y) ** 2)
        h = np.sqrt((p[0].x - p[3].x) ** 2 + (p[0].y - p[3].y) ** 2)
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    @staticmethod
    def _compute_iris_ratio(landmarks, eye_indices, iris_ring_indices) -> float:
        p0, p3 = landmarks[eye_indices[0]], landmarks[eye_indices[3]]
        eye_w = np.sqrt((p0.x - p3.x) ** 2 + (p0.y - p3.y) ** 2)
        if eye_w == 0:
            return 0.0
        il, ir = landmarks[iris_ring_indices[0]], landmarks[iris_ring_indices[2]]
        iris_w = np.sqrt((il.x - ir.x) ** 2 + (il.y - ir.y) ** 2)
        return iris_w / eye_w

    @staticmethod
    def _estimate_gaze(landmarks) -> str:
        eye_l, eye_r = landmarks[33], landmarks[133]
        iris_c = landmarks[468]
        ew = eye_r.x - eye_l.x
        if ew == 0:
            return "center"
        rx = (iris_c.x - eye_l.x) / ew
        if rx < 0.35:
            return "right"
        elif rx > 0.65:
            return "left"
        return "center"

    def release(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        print("[Vision] 리소스 해제 완료")