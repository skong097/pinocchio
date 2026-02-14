"""
피노키오 프로젝트 — 멀티모달 분석 엔진 v3.3 (Phase 3)
영상 6개 + 음성 7개 = 13개 지표 → 복합 스코어 + 7감정 분류

감정 분류: Blendshapes 기반 규칙 엔진
복합 스코어: 베이스라인 대비 변화율 가중 합산
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config.settings import SCORE_WEIGHTS, THRESHOLDS


# ──────────────────────────────────────────────
# 7감정 정의
# ──────────────────────────────────────────────
EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

EMOTION_LABELS_KR = {
    "happy": "행복 😊",
    "sad": "슬픔 😢",
    "angry": "분노 😠",
    "surprise": "놀람 😲",
    "fear": "불안 😰",
    "disgust": "불쾌 😖",
    "neutral": "보통 😐",
}


@dataclass
class AnalysisResult:
    """멀티모달 종합 분석 결과"""
    # 개별 스코어 (0~100)
    vision_score: float = 0.0
    voice_score: float = 0.0
    combined_score: float = 0.0

    # 감정 분류
    emotion: str = "neutral"
    emotion_kr: str = "보통 😐"
    emotion_confidence: float = 0.0
    emotion_scores: dict = field(default_factory=dict)

    # 스트레스 레벨
    stress_level: str = "low"
    anomaly_flags: list = field(default_factory=list)

    # 영상 지표 서브 스코어 (디버그용)
    sub_vision: dict = field(default_factory=dict)
    sub_voice: dict = field(default_factory=dict)


class Analyzer:
    """
    멀티모달 퓨전 분석기 v3.3
    영상 6개 + 음성 7개 = 13개 지표 → 복합 스코어 + 7감정 분류
    """

    def __init__(self):
        self.baseline_vision = {}
        self.baseline_voice = {}
        self._baseline_ready = False

    # ══════════════════════════════════════════
    # 메인 분석
    # ══════════════════════════════════════════

    def analyze(self, face_data, voice_data=None, blink_rate: float = 0.0,
                iris_instability: float = 0.0) -> AnalysisResult:
        """
        종합 분석

        Args:
            face_data: FaceData (vision.py)
            voice_data: VoiceData (voice_analyzer.py)
            blink_rate: 깜빡임/분 (VisionEngine.get_blink_rate())
            iris_instability: 홍채 불안정도 (VisionEngine.get_iris_instability())

        Returns:
            AnalysisResult
        """
        result = AnalysisResult()

        if not face_data.detected:
            return result

        # ── 1. 감정 분류 (Blendshapes 기반) ──
        emotion_scores = self._classify_emotion(face_data)
        result.emotion_scores = emotion_scores

        top_emotion = max(emotion_scores, key=emotion_scores.get)
        result.emotion = top_emotion
        result.emotion_kr = EMOTION_LABELS_KR.get(top_emotion, "보통 😐")
        result.emotion_confidence = emotion_scores[top_emotion]

        # ── 2. 영상 스코어 (6개 지표) ──
        vision_sub = self._compute_vision_score(face_data, blink_rate, iris_instability)
        result.sub_vision = vision_sub
        result.vision_score = self._weighted_sum(vision_sub, SCORE_WEIGHTS["vision"])

        # ── 3. 음성 스코어 (7개 지표) ──
        if voice_data and voice_data.f0_mean > 0:
            voice_sub = self._compute_voice_score(voice_data)
            result.sub_voice = voice_sub
            result.voice_score = self._weighted_sum(voice_sub, SCORE_WEIGHTS["voice"])

        # ── 4. 멀티모달 퓨전 ──
        fusion = SCORE_WEIGHTS["fusion"]
        if voice_data and voice_data.f0_mean > 0:
            result.combined_score = (
                result.vision_score * fusion["vision"]
                + result.voice_score * fusion["voice"]
            )
        else:
            # 음성 없으면 영상만 사용
            result.combined_score = result.vision_score

        result.combined_score = min(100, max(0, result.combined_score))

        # ── 5. 스트레스 레벨 ──
        if result.combined_score < THRESHOLDS["stress_low"]:
            result.stress_level = "low"
        elif result.combined_score < THRESHOLDS["stress_high"]:
            result.stress_level = "medium"
        else:
            result.stress_level = "high"

        # ── 6. 이상 감지 플래그 ──
        result.anomaly_flags = self._detect_anomalies(face_data, voice_data, blink_rate)

        return result

    # ══════════════════════════════════════════
    # 감정 분류 (Blendshapes 규칙 엔진)
    # ══════════════════════════════════════════

    def _classify_emotion(self, face_data) -> dict:
        """
        Blendshapes 기반 7감정 분류
        각 감정별 점수(0~1)를 반환, softmax 정규화
        """
        scores = {e: 0.0 for e in EMOTIONS}
        bs = face_data.blendshapes
        if not bs:
            scores["neutral"] = 1.0
            return scores

        # ── Happy ──
        smile_l = bs.get("mouthSmileLeft", 0)
        smile_r = bs.get("mouthSmileRight", 0)
        cheek_l = bs.get("cheekSquintLeft", 0)
        cheek_r = bs.get("cheekSquintRight", 0)
        scores["happy"] = (smile_l + smile_r) / 2 * 0.6 + (cheek_l + cheek_r) / 2 * 0.4

        # ── Sad ──
        frown_l = bs.get("mouthFrownLeft", 0)
        frown_r = bs.get("mouthFrownRight", 0)
        brow_inner = bs.get("browInnerUp", 0)
        scores["sad"] = (frown_l + frown_r) / 2 * 0.5 + brow_inner * 0.5

        # ── Angry ──
        brow_down_l = bs.get("browDownLeft", 0)
        brow_down_r = bs.get("browDownRight", 0)
        mouth_press_l = bs.get("mouthPressLeft", 0)
        mouth_press_r = bs.get("mouthPressRight", 0)
        jaw_clench = bs.get("jawForward", 0)
        scores["angry"] = (
            (brow_down_l + brow_down_r) / 2 * 0.4
            + (mouth_press_l + mouth_press_r) / 2 * 0.3
            + jaw_clench * 0.3
        )

        # ── Surprise ──
        eye_wide_l = bs.get("eyeWideLeft", 0)
        eye_wide_r = bs.get("eyeWideRight", 0)
        brow_outer_l = bs.get("browOuterUpLeft", 0)
        brow_outer_r = bs.get("browOuterUpRight", 0)
        jaw_open = bs.get("jawOpen", 0)
        scores["surprise"] = (
            (eye_wide_l + eye_wide_r) / 2 * 0.3
            + brow_inner * 0.2
            + (brow_outer_l + brow_outer_r) / 2 * 0.2
            + jaw_open * 0.3
        )

        # ── Fear ──
        scores["fear"] = (
            brow_inner * 0.3
            + (eye_wide_l + eye_wide_r) / 2 * 0.3
            + (mouth_press_l + mouth_press_r) / 2 * 0.2
            + face_data.lip_press_score * 0.2
        )

        # ── Disgust ──
        nose_l = bs.get("noseSneerLeft", 0)
        nose_r = bs.get("noseSneerRight", 0)
        upper_lip = bs.get("mouthShrugUpper", 0)
        scores["disgust"] = (
            (nose_l + nose_r) / 2 * 0.5
            + upper_lip * 0.3
            + (frown_l + frown_r) / 2 * 0.2
        )

        # ── Neutral ──
        # 다른 감정이 모두 낮으면 neutral
        max_others = max(scores[e] for e in EMOTIONS if e != "neutral")
        scores["neutral"] = max(0, 0.5 - max_others)

        # softmax 정규화
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}
        else:
            scores["neutral"] = 1.0

        return scores

    # ══════════════════════════════════════════
    # 영상 스코어 (6개 지표 → 0~100)
    # ══════════════════════════════════════════

    def _compute_vision_score(self, face_data, blink_rate, iris_instability) -> dict:
        """각 영상 지표를 0~100 스코어로 변환"""
        sub = {}

        # 1. 홍채 불안정도 (기존 stress_score)
        sub["iris_instability"] = min(100, iris_instability)

        # 2. 깜빡임 변화율
        lo, hi = THRESHOLDS["blink_normal_range"]
        mid = (lo + hi) / 2
        if blink_rate > 0:
            blink_dev = abs(blink_rate - mid) / mid * 100
            sub["blink_rate_change"] = min(100, blink_dev)
        else:
            sub["blink_rate_change"] = 0

        # 3. 동공 확장률
        sub["pupil_dilation"] = min(100, abs(face_data.pupil_dilation_pct) * 5)

        # 4. 미세 표정
        mc = face_data.micro_expression_count
        sub["micro_expression"] = min(100, mc * 20)  # 5건이면 100

        # 5. 비대칭
        asym = face_data.asymmetry_score
        sub["asymmetry"] = min(100, asym / THRESHOLDS["asymmetry_threshold"] * 50)

        # 6. 입술 압축
        lip = face_data.lip_press_score
        sub["lip_press"] = min(100, lip / THRESHOLDS["lip_press_threshold"] * 50)

        return sub

    # ══════════════════════════════════════════
    # 음성 스코어 (7개 지표 → 0~100)
    # ══════════════════════════════════════════

    def _compute_voice_score(self, voice_data) -> dict:
        """각 음성 지표를 0~100 스코어로 변환"""
        sub = {}

        # 1. 피치 변화 (F0 std 기반)
        sub["pitch_change"] = min(100, voice_data.f0_std * 2)

        # 2. 응답 지연
        lat = voice_data.response_latency_sec
        if lat > THRESHOLDS.get("response_latency_stress", 3.0):
            sub["response_latency"] = min(100, lat * 20)
        elif lat > THRESHOLDS.get("response_latency_normal", 2.0):
            sub["response_latency"] = min(100, lat * 15)
        else:
            sub["response_latency"] = max(0, lat * 5)

        # 3. 비유창성
        sub["disfluency"] = min(100, voice_data.disfluency_total * 25)

        # 4. Jitter
        sub["jitter_change"] = min(100, voice_data.jitter_pct * 10)

        # 5. 말 속도 변화
        if voice_data.speech_rate > 0:
            # 정상 한국어 말 속도: 3~5 음절/초
            rate_dev = abs(voice_data.speech_rate - 4.0) / 4.0 * 100
            sub["speech_rate_change"] = min(100, rate_dev)
        else:
            sub["speech_rate_change"] = 0

        # 6. 음량 변화
        sub["volume_change"] = min(100, voice_data.volume_std * 500)

        return sub

    # ══════════════════════════════════════════
    # 이상 감지
    # ══════════════════════════════════════════

    def _detect_anomalies(self, face_data, voice_data, blink_rate) -> list:
        """이상 패턴 감지 → 플래그 리스트"""
        flags = []

        # 시선 회피 과다
        if face_data.gaze_aversion_count > 8:
            flags.append("시선 회피 빈번")

        # 깜빡임 급증
        lo, hi = THRESHOLDS["blink_normal_range"]
        if blink_rate > hi * 1.5:
            flags.append("깜빡임 급증")

        # 미세 표정 다발
        if face_data.micro_expression_count >= 5:
            flags.append("미세 표정 다발")

        # 입술 압축 지속
        if face_data.lip_press_score > THRESHOLDS["lip_press_threshold"] * 1.5:
            flags.append("입술 압축 강함")

        # 음성 이상
        if voice_data and voice_data.f0_mean > 0:
            if voice_data.response_latency_sec > 5.0:
                flags.append("응답 지연 과다")
            if voice_data.disfluency_total >= 4:
                flags.append("비유창성 과다")

        return flags

    # ══════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════

    @staticmethod
    def _weighted_sum(sub_scores: dict, weights: dict) -> float:
        """가중 합산"""
        total = 0.0
        for key, weight in weights.items():
            total += sub_scores.get(key, 0) * weight
        return total

    def set_baseline(self, vision_baseline: dict, voice_baseline: dict = None):
        """베이스라인 설정"""
        self.baseline_vision = vision_baseline
        if voice_baseline:
            self.baseline_voice = voice_baseline
        self._baseline_ready = True
        print(f"[Analyzer] 베이스라인 설정 완료 (영상: {len(vision_baseline)}항목)")

    @property
    def is_baseline_ready(self) -> bool:
        return self._baseline_ready
