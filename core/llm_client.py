"""
피노키오 프로젝트 — LLM 클라이언트 v3.4 (Phase 4)
Ollama EXAONE 3.5 연동

Phase 4:
  - 다차원 프롬프트 (감정/영상지표/음성지표/이상 플래그)
  - 대화 전략 (워밍업 → 탐색 → 심화 → 요약)
  - 대화 히스토리 기반 맥락 유지
"""
import requests
import threading
from typing import Callable, Optional
from config.settings import LLM


# ──────────────────────────────────────────────
# 대화 전략 단계 정의
# ──────────────────────────────────────────────
STRATEGY = {
    "warmup": {
        "turns": (0, 2),         # 0~2턴
        "goal": "편안한 분위기 조성",
        "instruction": (
            "지금은 워밍업 단계입니다. "
            "가벼운 일상 질문으로 편안한 분위기를 만드세요. "
            "분석 결과는 아직 언급하지 마세요."
        ),
    },
    "explore": {
        "turns": (3, 6),         # 3~6턴
        "goal": "감정 탐색",
        "instruction": (
            "지금은 탐색 단계입니다. "
            "사용자의 감정과 최근 경험에 대해 자연스럽게 물어보세요. "
            "분석에서 감지된 감정 변화를 참고하되 직접 언급하지 마세요."
        ),
    },
    "deepen": {
        "turns": (7, 12),        # 7~12턴
        "goal": "심화 대화",
        "instruction": (
            "지금은 심화 단계입니다. "
            "이전 대화에서 나온 핵심 주제를 깊이 탐색하세요. "
            "이상 패턴이 감지되면 부드럽게 관련 질문을 하세요."
        ),
    },
    "summary": {
        "turns": (13, 999),      # 13턴 이후
        "goal": "대화 정리",
        "instruction": (
            "지금은 마무리 단계입니다. "
            "대화를 따뜻하게 정리하면서 긍정적인 메시지를 전하세요. "
            "필요하면 다음에 또 이야기하자고 마무리하세요."
        ),
    },
}


class LLMClient:
    """Ollama API 기반 LLM 클라이언트 v3.4"""

    SYSTEM_PROMPT = (
        "당신은 '피노키오'라는 심리 분석 AI입니다.\n"
        "성격: 친한 친구처럼 편안하고 따뜻합니다.\n"
        "말투: 짧은 경어체(~요, ~죠, ~네요). 딱딱한 존댓말 금지.\n"
        "규칙:\n"
        "- 공감 한 마디 + 질문 하나, 총 두 문장 이내\n"
        "- 분석 데이터를 직접 수치로 말하지 마세요 (자연스러운 관찰로)\n"
        "- 사용자가 불편해하면 주제를 바꾸세요\n"
        "- 한국어로만 답하세요"
    )

    def __init__(self):
        self.url = LLM["base_url"]
        self.model = LLM["model"]
        self.timeout = LLM["timeout"]
        self.conversation_history = []
        self._turn_count = 0

    # ══════════════════════════════════════════
    # API 호출
    # ══════════════════════════════════════════

    def generate(self, prompt: str) -> Optional[str]:
        try:
            res = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": LLM["stream"],
                    "options": {
                        "num_predict": LLM.get("num_predict", 80),
                        "temperature": LLM.get("temperature", 0.7),
                    },
                },
                timeout=self.timeout,
            )
            if res.status_code == 200:
                answer = res.json().get("response", "").strip()
                self.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                })
                self._turn_count += 1
                return answer
            else:
                print(f"[LLM] HTTP {res.status_code}: {res.text[:100]}")
                return None
        except requests.Timeout:
            print(f"[LLM] 타임아웃 ({self.timeout}초)")
            return None
        except requests.ConnectionError:
            print("[LLM] Ollama 서버 연결 실패 — ollama serve 확인")
            return None
        except Exception as e:
            print(f"[LLM] 에러: {e}")
            return None

    def generate_async(self, prompt: str, callback: Callable[[Optional[str]], None]):
        def _run():
            result = self.generate(prompt)
            callback(result)
        threading.Thread(target=_run, daemon=True).start()

    # ══════════════════════════════════════════
    # Phase 4: 다차원 프롬프트 생성
    # ══════════════════════════════════════════

    def build_prompt(self, stress_score: int, user_text: str,
                     analysis_result=None, face_data=None,
                     voice_data=None) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        # ── 전략 (한 단어) ──
        stage = self.get_strategy_name()

        # ── 핵심 관찰 (키워드만) ──
        obs = []
        if analysis_result:
            obs.append(analysis_result.emotion_kr)
            if analysis_result.anomaly_flags:
                obs.extend(analysis_result.anomaly_flags[:2])

        if face_data and face_data.detected:
            if face_data.gaze_aversion_count > 3:
                obs.append("시선회피")
            if face_data.micro_expression_count > 0:
                obs.append(face_data.micro_expression_label)

        if voice_data and voice_data.f0_mean > 0:
            if voice_data.response_latency_sec > 3.0:
                obs.append("응답지연")
            if voice_data.disfluency_total >= 2:
                obs.append("말더듬")

        obs_str = ", ".join(obs) if obs else "평온"

        # ── 최근 대화 (마지막 1턴만) ──
        last = ""
        for entry in reversed(self.conversation_history[:-1]):
            if entry["role"] == "assistant":
                last = entry["content"][:40]
                break

        # ── 프롬프트 조립 (최소화) ──
        prompt = (
            f"[{stage}] 관찰:{obs_str} 스트레스:{stress_score}\n"
        )
        if last:
            prompt += f"이전답변:{last}\n"
        prompt += f"사용자:{user_text}"

        return prompt

    # ══════════════════════════════════════════
    # 대화 전략
    # ══════════════════════════════════════════

    def _get_current_strategy(self) -> dict:
        """현재 턴에 해당하는 대화 전략 반환"""
        for stage_name, stage in STRATEGY.items():
            lo, hi = stage["turns"]
            if lo <= self._turn_count <= hi:
                return stage
        return STRATEGY["summary"]

    def get_strategy_name(self) -> str:
        """현재 전략 단계 이름"""
        for name, stage in STRATEGY.items():
            lo, hi = stage["turns"]
            if lo <= self._turn_count <= hi:
                return name
        return "summary"

    # ══════════════════════════════════════════
    # 대화 히스토리
    # ══════════════════════════════════════════

    def _get_recent_history(self, max_turns: int = 3) -> str:
        """최근 N턴 히스토리를 문자열로"""
        if len(self.conversation_history) < 2:
            return ""

        recent = self.conversation_history[-(max_turns * 2):]
        lines = []
        for entry in recent:
            role = "피노키오" if entry["role"] == "assistant" else "사용자"
            content = entry["content"][:60]  # 길면 잘라서
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def clear_history(self):
        self.conversation_history.clear()
        self._turn_count = 0
