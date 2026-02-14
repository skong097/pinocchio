"""
피노키오 프로젝트 — 음성 분석 모듈
librosa 기반 음성 특징 추출 (피치, Jitter/Shimmer, 말 속도, 비유창성 등)
Phase 2B에서 본격 활성화
"""
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config.settings import VOICE


@dataclass
class VoiceData:
    """한 발화의 음성 분석 결과"""
    # 피치 (F0)
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_max: float = 0.0
    f0_range: float = 0.0
    f0_raw: Optional[np.ndarray] = None

    # Jitter / Shimmer
    jitter_pct: float = 0.0
    shimmer_pct: float = 0.0

    # 응답 지연
    response_latency_sec: float = 0.0

    # 말 속도
    speech_rate: float = 0.0       # 음절/초
    duration_sec: float = 0.0

    # 음량
    volume_mean: float = 0.0
    volume_std: float = 0.0
    volume_range: float = 0.0

    # 비유창성
    filler_count: int = 0          # "음..", "어.." 횟수
    repetition_count: int = 0      # 단어 반복 횟수
    disfluency_total: int = 0


class VoiceAnalyzer:
    """
    음성 특징 추출기
    WAV 파일 + STT 텍스트를 받아 다차원 음성 지표 반환
    """

    def __init__(self):
        self.baseline = {}
        self._librosa = None
        self._load_librosa()

    def _load_librosa(self):
        """librosa 지연 로딩"""
        try:
            import librosa
            self._librosa = librosa
            print("[VoiceAnalyzer] librosa 로드 완료")
        except ImportError:
            print("[VoiceAnalyzer] librosa 미설치 — 음성 분석 비활성화")
            print("  → pip install librosa soundfile")

    def analyze(self, wav_path: str, text: str = "",
                response_latency: float = 0.0) -> VoiceData:
        """
        WAV 파일 종합 분석

        Args:
            wav_path: WAV 파일 경로
            text: STT 인식 텍스트 (비유창성 분석용)
            response_latency: AI 질문 끝 → 사용자 발화 시작 (초)

        Returns:
            VoiceData: 음성 분석 결과
        """
        voice_data = VoiceData(response_latency_sec=response_latency)

        if self._librosa is None:
            return voice_data

        try:
            y, sr = self._librosa.load(wav_path, sr=VOICE["sample_rate"])
        except Exception as e:
            print(f"[VoiceAnalyzer] WAV 로드 실패: {e}")
            return voice_data

        # 피치 분석
        self._analyze_pitch(y, sr, voice_data)

        # Jitter / Shimmer
        self._analyze_jitter(voice_data)
        self._analyze_shimmer(y, sr, voice_data)

        # 말 속도
        self._analyze_speech_rate(y, sr, text, voice_data)

        # 음량
        self._analyze_volume(y, sr, voice_data)

        # 비유창성 (텍스트 기반)
        self._detect_disfluency(text, voice_data)

        return voice_data

    def _analyze_pitch(self, y, sr, vd: VoiceData):
        """YIN 알고리즘으로 F0 추출"""
        try:
            f0 = self._librosa.yin(
                y, fmin=VOICE["f0_min"], fmax=VOICE["f0_max"], sr=sr
            )
            f0_valid = f0[f0 > 0]
            if len(f0_valid) > 0:
                vd.f0_mean = float(np.mean(f0_valid))
                vd.f0_std = float(np.std(f0_valid))
                vd.f0_max = float(np.max(f0_valid))
                vd.f0_range = float(np.ptp(f0_valid))
                vd.f0_raw = f0
        except Exception as e:
            print(f"[VoiceAnalyzer] 피치 분석 실패: {e}")

    def _analyze_jitter(self, vd: VoiceData):
        """연속 피치 주기 간 변동률"""
        if vd.f0_raw is None:
            return
        f0_valid = vd.f0_raw[vd.f0_raw > 0]
        if len(f0_valid) < 2:
            return
        periods = 1.0 / f0_valid
        mean_period = np.mean(periods)
        if mean_period > 0:
            vd.jitter_pct = float(
                np.mean(np.abs(np.diff(periods))) / mean_period * 100
            )

    def _analyze_shimmer(self, y, sr, vd: VoiceData):
        """진폭 변동률 (RMS 기반)"""
        try:
            rms = self._librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            if mean_rms > 0:
                vd.shimmer_pct = float(
                    np.mean(np.abs(np.diff(rms))) / mean_rms * 100
                )
        except Exception as e:
            print(f"[VoiceAnalyzer] Shimmer 분석 실패: {e}")

    def _analyze_speech_rate(self, y, sr, text: str, vd: VoiceData):
        """말 속도 (음절/초)"""
        duration = self._librosa.get_duration(y=y, sr=sr)
        vd.duration_sec = float(duration)
        if duration > 0 and text:
            syllable_count = len(text)  # 한국어: 글자수 ≈ 음절수
            vd.speech_rate = float(syllable_count / duration)

    def _analyze_volume(self, y, sr, vd: VoiceData):
        """음량 통계"""
        try:
            rms = self._librosa.feature.rms(y=y)[0]
            vd.volume_mean = float(np.mean(rms))
            vd.volume_std = float(np.std(rms))
            vd.volume_range = float(np.ptp(rms))
        except Exception as e:
            print(f"[VoiceAnalyzer] 음량 분석 실패: {e}")

    @staticmethod
    def _detect_disfluency(text: str, vd: VoiceData):
        """STT 텍스트에서 비유창성 패턴 감지"""
        if not text:
            return
        fillers = re.findall(r'(음+|어+|아+|그+|저+)', text)
        repetitions = re.findall(r'(\b\w+\b)(?:\s+\1)+', text)
        vd.filler_count = len(fillers)
        vd.repetition_count = len(repetitions)
        vd.disfluency_total = vd.filler_count + vd.repetition_count
