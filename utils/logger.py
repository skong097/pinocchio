"""
피노키오 프로젝트 — 데이터 로거
세션 데이터를 JSON으로 저장 (분석 리포트용)
"""
import json
import time
from datetime import datetime
from pathlib import Path
from config.settings import LOG_DIR


class SessionLogger:
    """세션 데이터 JSON 로거"""

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"session_{self.session_id}.json"
        self.entries = []
        self._start_time = time.time()
        print(f"[Logger] 세션 시작: {self.session_id}")

    def log_frame(self, face_data=None, voice_data=None,
                  analysis_result=None):
        """프레임 단위 데이터 기록"""
        entry = {
            "timestamp": time.time(),
            "elapsed_sec": time.time() - self._start_time,
        }
        if face_data and face_data.detected:
            entry["vision"] = {
                "ear_left": face_data.ear_left,
                "ear_right": face_data.ear_right,
                "iris_ratio_left": face_data.iris_ratio_left,
                "iris_ratio_right": face_data.iris_ratio_right,
                "gaze_direction": face_data.gaze_direction,
                "is_blinking": face_data.is_blinking,
            }
        if voice_data:
            entry["voice"] = {
                "f0_mean": voice_data.f0_mean,
                "speech_rate": voice_data.speech_rate,
                "disfluency_total": voice_data.disfluency_total,
                "response_latency_sec": voice_data.response_latency_sec,
            }
        if analysis_result:
            entry["analysis"] = {
                "vision_score": analysis_result.vision_score,
                "voice_score": analysis_result.voice_score,
                "combined_score": analysis_result.combined_score,
                "stress_level": analysis_result.stress_level,
            }
        self.entries.append(entry)

    def log_conversation(self, role: str, text: str):
        """대화 기록"""
        self.entries.append({
            "timestamp": time.time(),
            "elapsed_sec": time.time() - self._start_time,
            "conversation": {
                "role": role,
                "text": text,
            }
        })

    def save(self):
        """세션 데이터 파일 저장"""
        data = {
            "session_id": self.session_id,
            "start_time": self._start_time,
            "duration_sec": time.time() - self._start_time,
            "total_entries": len(self.entries),
            "entries": self.entries,
        }
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Logger] 세션 저장: {self.log_file}")
