# 🤥 피노키오 (EyeCon) v3.4

**실시간 멀티모달 심리 상태 분석 AI**

MediaPipe 478점 FaceLandmarker + 52 Blendshapes 기반의 영상 분석과 librosa 음성 분석을 결합하여 13개 심리 지표를 실시간 추출하고, 7가지 감정을 분류하며, Ollama EXAONE LLM과 자연스러운 한국어 대화를 수행하는 심리 분석 AI 시스템입니다.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-orange)
![Ollama](https://img.shields.io/badge/LLM-EXAONE_3.5-red)

---

## 주요 기능

### 영상 분석 (6개 지표)
- **홍채 추적** — 시선 방향 + 불안정도 + 고정/회피 패턴 (60초 윈도우)
- **깜빡임 분석** — EAR 기반 실시간 깜빡임/분 추적 (정상 범위: 15~20회)
- **동공 확장률** — 최초 10초 자동 베이스라인 → 변화율(%) 실시간 비교
- **미세 표정 감지** — Blendshapes 급변(0.5초 내 delta>0.3) 유형 라벨링
- **표정 비대칭** — 6개 좌/우 AU 쌍 종합 비대칭 점수
- **입술 압축** — AU24 기반 거짓말 지표

### 음성 분석 (7개 지표)
- **피치 (F0)** — librosa YIN 알고리즘 (75~500Hz)
- **Jitter / Shimmer** — 음성 떨림 지표
- **응답 지연** — TTS 종료 → 사용자 발화 시작 시간 차이
- **말 속도** — 음절/초 (한국어 기준)
- **음량 변화** — RMS 기반 통계
- **비유창성** — "음", "어" 등 필러 + 단어 반복 감지

### 7감정 분류
Blendshapes 규칙 엔진 기반 실시간 감정 분류:
Happy 😊 · Sad 😢 · Angry 😠 · Surprise 😲 · Fear 😰 · Disgust 😖 · Neutral 😐

### 멀티모달 복합 스코어
- 영상 60% + 음성 40% 가중 퓨전
- 0~100 스트레스 점수 + 이상 감지 플래그

### LLM 대화 (Phase 4)
- Ollama EXAONE 3.5 (7.8B) 한국어 대화
- 4단계 대화 전략 (워밍업 → 탐색 → 심화 → 요약)
- 다차원 프롬프트 (감정/지표/이상 플래그 실시간 반영)
- Edge TTS 한국어 음성 합성

---

## 데모 화면

```
┌──────────────────┬──────────────────────────┐
│   📷 실시간 영상  │  😊 행복 (감정 분류)       │
│  (썬글라스 오버레이│  ████████░░ 종합: 42     │
│   + 홍채 추적)    │  영상 지표 / 음성 지표     │
├──────────────────┼──────────────────────────┤
│  📊 실시간 그래프  │  💬 대화 로그              │
│  (스트레스/깜빡임) │  [피노키오] / [사용자]     │
└──────────────────┴──────────────────────────┘
```

---

## 프로젝트 구조

```
eyecon/
├── main.py                      # 메인 윈도우 (PyQt6 4분할 대시보드)
├── config/
│   ├── __init__.py
│   └── settings.py              # 설정값 중앙 관리 (경로/임계값/가중치)
├── core/
│   ├── __init__.py
│   ├── vision.py                # MediaPipe FaceLandmarker 영상 분석 엔진
│   ├── voice_analyzer.py        # librosa 음성 특징 추출기
│   ├── analyzer.py              # 멀티모달 퓨전 분석 + 7감정 분류
│   ├── baseline.py              # 개인별 베이스라인 수집기 (30초)
│   └── llm_client.py            # Ollama EXAONE LLM 클라이언트
├── utils/
│   ├── __init__.py
│   ├── tts_engine.py            # Edge TTS 음성 합성
│   ├── stt_engine.py            # Google STT 음성 인식
│   └── logger.py                # 세션 로그 (JSON)
├── assets/
│   └── face_landmarker_v2_with_blendshapes.task
├── logs/                        # 세션별 대화 로그
├── temp/                        # WAV/MP3 임시 파일
├── check_mic.py                 # 마이크 진단 도구
└── requirements.txt
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상 환경 생성
python -m venv eyecon_venv
source eyecon_venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Ollama + EXAONE 모델 설치

```bash
# Ollama 설치 (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# EXAONE 3.5 모델 다운로드
ollama pull exaone3.5:7.8b

# Ollama 서버 실행
ollama serve
```

### 3. MediaPipe 모델

`assets/` 디렉토리에 `face_landmarker_v2_with_blendshapes.task` 파일이 필요합니다.
[Google MediaPipe 공식 페이지](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)에서 다운로드하세요.

### 4. 실행

```bash
cd eyecon
python main.py
```

---

## 설정 커스터마이징

`config/settings.py`에서 주요 설정을 조정할 수 있습니다:

### 마이크 설정
```python
STT = {
    "microphone_index": 10,        # PipeWire 장치 (None=시스템 기본)
    "energy_threshold": 15000,     # 에너지 임계값 (내장 DMIC 기준)
    "dynamic_energy": False,       # 수동 고정
    "pause_threshold": 1.0,        # 발화 종료 판단 (초)
    "phrase_time_limit": 15,       # 최대 녹음 시간 (초)
}
```

### LLM 설정
```python
LLM = {
    "model": "exaone3.5",         # Ollama 모델명
    "num_predict": 80,             # 최대 생성 토큰
    "temperature": 0.7,
}
```

### 마이크 진단
```bash
# 장치 목록 확인 + 테스트
python check_mic.py

# 특정 장치 적용
python check_mic.py --apply 10
```

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| 영상 분석 | MediaPipe FaceLandmarker (478점 + 52 Blendshapes) |
| 음성 분석 | librosa (YIN F0, RMS, Jitter/Shimmer) |
| 감정 분류 | Blendshapes 규칙 엔진 (7감정) |
| LLM | Ollama + EXAONE 3.5 (7.8B) |
| TTS | Edge TTS (ko-KR-SunHiNeural) |
| STT | Google Speech Recognition |
| GUI | PyQt6 + pyqtgraph |
| 언어 | Python 3.12 |

---

## 아키텍처

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  WebCam      │───▶│  VisionEngine │───▶│             │
│  (30fps)     │    │  (MediaPipe)  │    │  Analyzer   │
└─────────────┘    └──────────────┘    │  (Fusion)   │
                                        │             │──▶ 복합 스코어
┌─────────────┐    ┌──────────────┐    │  영상 60%   │    + 감정 분류
│  Microphone  │───▶│  STT Engine   │───▶│  음성 40%   │    + 이상 플래그
│  (PipeWire)  │    │  (Google)     │    │             │
└─────────────┘    └──────┬───────┘    └──────┬──────┘
                          │                    │
                   ┌──────▼───────┐    ┌──────▼──────┐
                   │ VoiceAnalyzer │    │  LLMClient  │
                   │  (librosa)    │    │  (EXAONE)   │
                   └──────────────┘    └──────┬──────┘
                                              │
                                       ┌──────▼──────┐
                                       │  TTS Engine  │
                                       │  (Edge TTS)  │
                                       └─────────────┘
```

---

## 최적화

응답 지연을 최소화하기 위한 최적화가 적용되어 있습니다:

- **프롬프트 압축** — 시스템 프롬프트 분리 + 데이터 최소화 (~80자)
- **병렬 처리** — 음성 분석과 LLM 추론을 동시 실행
- **프레임 경량화** — 영상만 매 프레임 분석, 음성은 이벤트 기반
- **토큰 제한** — `num_predict: 80`으로 응답 길이 제한

체감 응답 지연: 약 1.5초 (STT 종료 → TTS 시작)

---

## 버전 이력

| 버전 | 날짜 | 내용 |
|------|------|------|
| v3.4 | 2026-02-14 | Phase 4: LLM 다차원 프롬프트 + 대화 전략 + 최적화 튜닝 |
| v3.3 | 2026-02-14 | Phase 3: 7감정 분류 + 멀티모달 복합 스코어 + 베이스라인 |
| v3.2 | 2026-02-14 | Phase 2: 영상 6개 + 음성 7개 지표 + 마이크 설정 |
| v3.1 | 2026-02-14 | Phase 1: 4분할 대시보드 + pyqtgraph + 대화 루프 |
| v3.0 | 2026-02-13 | 7개 모듈 분리 + FaceLandmarker 전환 |
| v2.6 | 2026-02-12 | 원본 단일 파일 (EyeCon) |

---

## 라이선스

이 프로젝트는 개인 연구/학습 목적으로 개발되었습니다.

---

## 참고

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [EXAONE 3.5](https://huggingface.co/LGAI-EXAONE)
- [Ollama](https://ollama.com)
- [librosa](https://librosa.org)
- [Edge TTS](https://github.com/rany2/edge-tts)
