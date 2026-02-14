"""
피노키오 프로젝트 — TTS 엔진
Edge TTS 기반 한국어 음성 출력
"""
import asyncio
import threading
import time
from typing import Callable, Optional
from config.settings import TTS, TEMP_DIR

# pygame 지연 로딩
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class TTSEngine:
    """Edge TTS + pygame 기반 음성 출력"""

    def __init__(self):
        self.voice = TTS["voice"]
        self.output_file = str(TTS["output_file"])
        self._speaking = False
        self._speak_end_time = 0.0

        # temp 디렉토리 생성
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # pygame mixer 초기화
        if PYGAME_AVAILABLE:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            print(f"[TTS] Edge TTS 초기화 완료 (voice: {self.voice})")
        else:
            print("[TTS] pygame 미설치 — pip install pygame")

    async def _speak_async(self, text: str):
        """비동기 TTS 실행"""
        if not EDGE_TTS_AVAILABLE:
            print("[TTS] edge-tts 미설치 — pip install edge-tts")
            return

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(self.output_file)

            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(self.output_file)
                pygame.mixer.music.play()
                self._speaking = True

                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)

                pygame.mixer.music.unload()
                self._speaking = False
                self._speak_end_time = time.time()
        except Exception as e:
            print(f"[TTS] 에러: {e}")
            self._speaking = False

    def speak(self, text: str, on_complete: Optional[Callable] = None):
        """
        별도 스레드에서 TTS 실행

        Args:
            text: 발화 텍스트
            on_complete: 발화 완료 후 콜백
        """
        def _run():
            asyncio.run(self._speak_async(text))
            if on_complete:
                on_complete()

        threading.Thread(target=_run, daemon=True).start()

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def last_speak_end_time(self) -> float:
        """마지막 발화 종료 시간 (응답 지연 계산용)"""
        return self._speak_end_time

    def release(self):
        """리소스 해제"""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
        print("[TTS] 리소스 해제 완료")
