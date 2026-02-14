"""
피노키오 프로젝트 — STT 엔진
Google Speech Recognition + WAV 저장 (음성 분석용)
마이크 장치 선택 + 에너지 임계값 수동 고정 지원
"""
import time
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, QThread
from config.settings import STT, TEMP_DIR

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class STTWorker(QThread):
    """
    음성 인식 전용 QThread
    - STT 텍스트 결과 → result_signal
    - WAV 파일 경로 → audio_signal (음성 분석용)
    - 발화 시작 시간 → timing_signal (응답 지연 계산용)
    """
    result_signal = pyqtSignal(str)
    audio_signal = pyqtSignal(str)
    timing_signal = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self._wav_path = str(TEMP_DIR / "last_response.wav")

    def run(self):
        if not SR_AVAILABLE:
            print("[STT] SpeechRecognition 미설치")
            self.result_signal.emit("(음성 인식 불가)")
            return

        recognizer = sr.Recognizer()

        # ── 에너지 임계값 설정 ──
        if not STT.get("dynamic_energy", True):
            recognizer.energy_threshold = STT.get("energy_threshold", 5000)
            recognizer.dynamic_energy_threshold = False
            print(f"[STT] 에너지 임계값 고정: {recognizer.energy_threshold}")
        
        recognizer.pause_threshold = STT.get("pause_threshold", 1.5)

        # ── 마이크 장치 선택 ──
        mic_index = STT.get("microphone_index", None)
        mic_kwargs = {"device_index": mic_index} if mic_index is not None else {}
        
        if mic_index is not None:
            print(f"[STT] 마이크 장치 [{mic_index}] 사용")
        else:
            print("[STT] 시스템 기본 마이크 사용")

        try:
            with sr.Microphone(**mic_kwargs) as source:
                # dynamic_energy가 True일 때만 배경 소음 보정
                if STT.get("dynamic_energy", True):
                    recognizer.adjust_for_ambient_noise(
                        source, duration=STT["ambient_noise_duration"]
                    )
                    print(f"[STT] 자동 임계값: {recognizer.energy_threshold:.0f}")

                print("[STT] 마이크 입력 대기 중...")
                audio = recognizer.listen(
                    source,
                    timeout=STT["timeout"],
                    phrase_time_limit=STT["phrase_time_limit"],
                )

                # 발화 시작 시간
                self.timing_signal.emit(time.time())

                # WAV 저장
                with open(self._wav_path, "wb") as f:
                    f.write(audio.get_wav_data())
                self.audio_signal.emit(self._wav_path)

                # STT 변환
                text = recognizer.recognize_google(
                    audio, language=STT["language"]
                )
                print(f"[STT] 인식: {text}")
                self.result_signal.emit(text)

        except sr.WaitTimeoutError:
            print("[STT] 타임아웃 — 입력 없음")
            self.result_signal.emit("(대답 없음)")
        except sr.UnknownValueError:
            print("[STT] 음성 인식 실패 — 다시 말해주세요")
            self.result_signal.emit("(인식 불가)")
        except Exception as e:
            print(f"[STT] 에러: {e}")
            self.result_signal.emit("(대답 없음)")
