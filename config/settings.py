"""
피노키오 프로젝트 (EyeCon) — 설정값 중앙 관리
경로, 임계값, 가중치 등 모든 설정을 한 곳에서 관리
"""
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정 (상대 경로 기반)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LOG_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

# MediaPipe 모델 경로
FACE_LANDMARKER_MODEL = ASSETS_DIR / "face_landmarker_v2_with_blendshapes.task"

# ──────────────────────────────────────────────
# 카메라 설정
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_FPS = 30  # ms 단위 타이머 간격
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# ──────────────────────────────────────────────
# MediaPipe Face Landmarker 설정
# ──────────────────────────────────────────────
FACE_LANDMARKER = {
    "max_num_faces": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "output_blendshapes": True,
    "output_face_transformation_matrixes": False,
}

# ──────────────────────────────────────────────
# 홍채 랜드마크 인덱스 (478점 기준)
# ──────────────────────────────────────────────
IRIS_LANDMARKS = {
    "left_center": 468,
    "left_ring": [469, 470, 471, 472],
    "right_center": 473,
    "right_ring": [474, 475, 476, 477],
}

# 눈 랜드마크 (EAR 계산용)
EYE_LANDMARKS = {
    "left": [33, 160, 158, 133, 153, 144],   # P1~P6
    "right": [362, 385, 387, 263, 373, 380],  # P1~P6
}

# ──────────────────────────────────────────────
# Blendshapes → AU 매핑 (미세 표정 분석용)
# ──────────────────────────────────────────────
BLENDSHAPE_AU_MAP = {
    # 눈썹
    "browDownLeft": "AU4_L",        # 미간 찌푸림 좌
    "browDownRight": "AU4_R",       # 미간 찌푸림 우
    "browInnerUp": "AU1",           # 내측 눈썹 올림
    "browOuterUpLeft": "AU2_L",     # 외측 눈썹 올림 좌
    "browOuterUpRight": "AU2_R",    # 외측 눈썹 올림 우
    # 눈
    "eyeBlinkLeft": "AU45_L",      # 눈 깜빡임 좌
    "eyeBlinkRight": "AU45_R",     # 눈 깜빡임 우
    "eyeSquintLeft": "AU6_L",      # 볼 올림 좌
    "eyeSquintRight": "AU6_R",     # 볼 올림 우
    "eyeWideLeft": "AU5_L",        # 눈 크게 뜸 좌
    "eyeWideRight": "AU5_R",       # 눈 크게 뜸 우
    # 입
    "jawOpen": "AU26",              # 턱 벌림
    "mouthSmileLeft": "AU12_L",     # 웃음 좌
    "mouthSmileRight": "AU12_R",    # 웃음 우
    "mouthFrownLeft": "AU15_L",     # 입꼬리 내림 좌
    "mouthFrownRight": "AU15_R",    # 입꼬리 내림 우
    "mouthPressLeft": "AU24_L",     # 입술 압축 좌 ← 거짓말 지표
    "mouthPressRight": "AU24_R",    # 입술 압축 우 ← 거짓말 지표
    "mouthPucker": "AU18",          # 입술 오므림
    "mouthShrugUpper": "AU10",      # 윗입술 올림
}

# ──────────────────────────────────────────────
# 분석 임계값
# ──────────────────────────────────────────────
THRESHOLDS = {
    # 깜빡임 감지
    "ear_blink": 0.20,              # EAR 이 값 이하이면 깜빡임
    "blink_consec_frames": 2,       # 연속 프레임 수
    "blink_normal_range": (15, 20), # 정상 깜빡임 범위 (회/분)

    # 동공 확장
    "pupil_dilation_threshold": 0.04,  # 4% 이상 변화 감지

    # 미세 표정
    "micro_expr_duration": 0.5,     # 0.5초 미만
    "micro_expr_delta": 0.3,        # blendshape 변화량 임계값

    # 표정 비대칭
    "asymmetry_threshold": 0.15,    # 좌우 차이 임계값

    # 시선
    "gaze_fixation_threshold": 2.0, # 초 — 고정 시간 임계값
    "iris_history_size": 30,        # 홍채 히스토리 프레임 수

    # 입술 압축
    "lip_press_threshold": 0.3,     # AU24 임계값

    # 스트레스 판정
    "stress_low": 25,
    "stress_high": 60,
}

# ──────────────────────────────────────────────
# 베이스라인 설정
# ──────────────────────────────────────────────
BASELINE = {
    "warmup_duration_sec": 30,      # 베이스라인 수집 시간
    "min_samples": 100,             # 최소 샘플 수
}

# ──────────────────────────────────────────────
# 음성 분석 설정
# ──────────────────────────────────────────────
VOICE = {
    "sample_rate": 16000,
    "f0_min": 75,
    "f0_max": 500,
    "response_latency_normal": 2.0,  # 정상 응답 지연 (초)
    "response_latency_stress": 3.0,  # 스트레스 응답 지연 (초)
}

# ──────────────────────────────────────────────
# 복합 스코어 가중치 (Phase 3에서 사용)
# ──────────────────────────────────────────────
SCORE_WEIGHTS = {
    # 영상 지표 가중치
    "vision": {
        "iris_instability": 0.15,
        "blink_rate_change": 0.15,
        "pupil_dilation": 0.20,
        "micro_expression": 0.20,
        "asymmetry": 0.15,
        "lip_press": 0.15,
    },
    # 음성 지표 가중치
    "voice": {
        "pitch_change": 0.20,
        "response_latency": 0.20,
        "disfluency": 0.15,
        "jitter_change": 0.15,
        "speech_rate_change": 0.15,
        "volume_change": 0.15,
    },
    # 멀티모달 퓨전 비율
    "fusion": {
        "vision": 0.6,
        "voice": 0.4,
    },
}

# ──────────────────────────────────────────────
# LLM 설정 (Ollama + EXAONE)
# ──────────────────────────────────────────────
LLM = {
    "base_url": "http://localhost:11434/api/generate",
    "model": "exaone3.5",
    "timeout": 15,
    "stream": False,
    "num_predict": 80,    # 최대 생성 토큰 수 (짧은 응답 유도)
    "temperature": 0.7,
}

# ──────────────────────────────────────────────
# TTS / STT 설정
# ──────────────────────────────────────────────
TTS = {
    "voice": "ko-KR-SunHiNeural",
    "output_file": TEMP_DIR / "response.mp3",
}

STT = {
    "language": "ko-KR",
    "timeout": 20,
    "phrase_time_limit": 15,           # 60 → 15 (핵심!)
    "ambient_noise_duration": 0.5,
    "microphone_index": 10,
    "energy_threshold": 15000,         # 5000 → 15000 (핵심!)
    "dynamic_energy": False,
    "pause_threshold": 1.0,            # 1.5초 침묵이면 발화 종료
}

# ──────────────────────────────────────────────
# UI 설정
# ──────────────────────────────────────────────
UI = {
    "window_title": "피노키오 v3.0 — 심리 상태 분석기",
    "window_geometry": (100, 100, 1100, 850),
    "colors": {
        "bg_dark": "#1a1a1a",
        "text_white": "#ffffff",
        "status_ok": "#00ff00",
        "status_warn": "#ffaa00",
        "status_alert": "#ff4444",
        "ai_text": "#00ff00",
        "user_text": "#ffffff",
        "system_text": "#55aaff",
    },
    "font_family": "Malgun Gothic",
    "font_size": 14,
}
