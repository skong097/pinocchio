"""
피노키오 프로젝트 — 베이스라인 학습 모듈 v3.3 (Phase 3)
대화 시작 후 30초간 정상 상태 수집 → 개인별 평균/표준편차 저장
Phase 2A 지표 포함 (동공확장률, 미세표정, 비대칭, 입술압축)
"""
import time
import numpy as np
from collections import deque
from config.settings import BASELINE


class BaselineCollector:
    """개인별 정상 상태 베이스라인 수집기"""

    def __init__(self):
        self.start_time = None
        self.vision_samples = deque()
        self.voice_samples = deque()
        self._ready = False
        self._on_complete_callback = None

    def start(self, on_complete=None):
        """
        베이스라인 수집 시작

        Args:
            on_complete: 수집 완료 시 콜백 (vision_baseline, voice_baseline)
        """
        self.start_time = time.time()
        self.vision_samples.clear()
        self.voice_samples.clear()
        self._ready = False
        self._on_complete_callback = on_complete
        print(f"[Baseline] 수집 시작 ({BASELINE['warmup_duration_sec']}초)")

    def add_vision_sample(self, face_data, blink_rate: float = 0.0,
                          iris_instability: float = 0.0):
        """영상 샘플 추가 (Phase 2A 지표 포함)"""
        if self._ready or not self.start_time:
            return

        self.vision_samples.append({
            "ear_avg": (face_data.ear_left + face_data.ear_right) / 2,
            "iris_ratio_avg": (face_data.iris_ratio_left + face_data.iris_ratio_right) / 2,
            "iris_instability": iris_instability,
            "blink_rate": blink_rate,
            "asymmetry": face_data.asymmetry_score,
            "lip_press": face_data.lip_press_score,
        })

        self._check_ready()

    def add_voice_sample(self, voice_data):
        """음성 샘플 추가"""
        if self._ready or not self.start_time:
            return

        if voice_data.f0_mean <= 0:
            return

        self.voice_samples.append({
            "f0_mean": voice_data.f0_mean,
            "f0_std": voice_data.f0_std,
            "speech_rate": voice_data.speech_rate,
            "volume_mean": voice_data.volume_mean,
            "jitter_pct": voice_data.jitter_pct,
        })

        self._check_ready()

    def _check_ready(self):
        """수집 완료 확인"""
        elapsed = time.time() - self.start_time
        if elapsed >= BASELINE["warmup_duration_sec"]:
            if len(self.vision_samples) >= BASELINE["min_samples"]:
                self._ready = True
                print(f"[Baseline] 수집 완료: "
                      f"영상 {len(self.vision_samples)}샘플, "
                      f"음성 {len(self.voice_samples)}샘플")

                # 콜백 실행
                if self._on_complete_callback:
                    self._on_complete_callback(
                        self.get_vision_baseline(),
                        self.get_voice_baseline(),
                    )

    def get_vision_baseline(self) -> dict:
        """영상 베이스라인 통계 (평균 + 표준편차)"""
        if not self.vision_samples:
            return {}
        keys = self.vision_samples[0].keys()
        result = {}
        for key in keys:
            values = [s[key] for s in self.vision_samples]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
        return result

    def get_voice_baseline(self) -> dict:
        """음성 베이스라인 통계"""
        if not self.voice_samples:
            return {}
        keys = self.voice_samples[0].keys()
        result = {}
        for key in keys:
            values = [s[key] for s in self.voice_samples]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
        return result

    def get_progress(self) -> float:
        """수집 진행률 (0~1)"""
        if not self.start_time:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / BASELINE["warmup_duration_sec"])

    @property
    def is_ready(self) -> bool:
        return self._ready
