"""
피노키오 프로젝트 — 마이크 진단 & 선택 도구
사용법: python check_mic.py
"""
import speech_recognition as sr
import pyaudio


def list_microphones():
    """시스템에 연결된 모든 마이크 목록 출력"""
    print("=" * 60)
    print("🎤 시스템 마이크 목록")
    print("=" * 60)

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    input_devices = []
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info))
            default_mark = " ⭐ (기본)" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {device_info['name']}"
                  f"  (채널: {device_info['maxInputChannels']}, "
                  f"SR: {int(device_info['defaultSampleRate'])}Hz)"
                  f"{default_mark}")

    p.terminate()
    print(f"\n총 {len(input_devices)}개 입력 장치 발견")
    print("=" * 60)
    return input_devices


def list_sr_microphones():
    """SpeechRecognition 라이브러리가 인식하는 마이크 목록"""
    print("\n🔍 SpeechRecognition 마이크 목록")
    print("-" * 60)
    mic_list = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_list):
        print(f"  [{i}] {name}")
    print(f"\n총 {len(mic_list)}개")
    return mic_list


def test_microphone(device_index=None):
    """특정 마이크로 녹음 테스트"""
    label = f"장치 [{device_index}]" if device_index is not None else "기본 장치"
    print(f"\n🎙️ 마이크 테스트: {label}")
    print("  → 3초간 아무 말이나 해보세요...")

    recognizer = sr.Recognizer()
    try:
        mic_args = {"device_index": device_index} if device_index is not None else {}
        with sr.Microphone(**mic_args) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("  → 듣는 중...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print(f"  → 녹음 완료! (데이터 크기: {len(audio.get_wav_data())} bytes)")

            # STT 테스트
            try:
                text = recognizer.recognize_google(audio, language='ko-KR')
                print(f"  ✅ 인식 결과: '{text}'")
                return True
            except sr.UnknownValueError:
                print("  ⚠️ 소리는 감지되었으나 음성 인식 실패 (더 크게 말해보세요)")
                return True  # 마이크 자체는 동작
            except sr.RequestError as e:
                print(f"  ❌ Google STT 오류: {e}")
                return True  # 마이크는 동작, 네트워크 문제

    except sr.WaitTimeoutError:
        print("  ❌ 타임아웃 — 소리가 감지되지 않았습니다")
        return False
    except OSError as e:
        print(f"  ❌ 장치 열기 실패: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 에러: {e}")
        return False


def main():
    # 1. 장치 목록
    devices = list_microphones()
    sr_mics = list_sr_microphones()

    # 2. 기본 마이크 테스트
    print("\n" + "=" * 60)
    print("📋 마이크 테스트")
    print("=" * 60)

    test_microphone(None)  # 기본 장치

    # 3. 사용자 선택
    while True:
        print("\n" + "-" * 60)
        choice = input("테스트할 장치 번호 입력 (q=종료, a=전체 테스트): ").strip()

        if choice.lower() == 'q':
            break
        elif choice.lower() == 'a':
            for idx, _ in devices:
                test_microphone(idx)
        elif choice.isdigit():
            idx = int(choice)
            success = test_microphone(idx)
            if success:
                apply = input(f"\n  이 장치 [{idx}]를 피노키오에 적용할까요? (y/n): ").strip()
                if apply.lower() == 'y':
                    print(f"\n  ✅ config/settings.py에 다음을 추가하세요:")
                    print(f'     MICROPHONE_INDEX = {idx}')
                    print(f"\n  그리고 utils/stt_engine.py의 sr.Microphone()을:")
                    print(f"     sr.Microphone(device_index={idx})")
                    print(f"  으로 변경하세요. 또는 아래 자동 적용을 실행:")
                    print(f"\n     python check_mic.py --apply {idx}")
        else:
            print("  잘못된 입력")

    print("\n종료!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--apply":
        idx = int(sys.argv[2])
        print(f"장치 [{idx}] 적용 중...")

        # settings.py에 MICROPHONE_INDEX 추가/수정
        settings_path = "config/settings.py"
        with open(settings_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "MICROPHONE_INDEX" in content:
            import re
            content = re.sub(
                r'MICROPHONE_INDEX\s*=\s*\S+',
                f'MICROPHONE_INDEX = {idx}',
                content
            )
        else:
            content = content.replace(
                'CAMERA_INDEX = 0',
                f'CAMERA_INDEX = 0\nMICROPHONE_INDEX = {idx}  # 마이크 장치 인덱스'
            )
        with open(settings_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ {settings_path} 업데이트 완료")

        # stt_engine.py 수정
        stt_path = "utils/stt_engine.py"
        with open(stt_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "device_index" not in content:
            content = content.replace(
                "from config.settings import STT, TEMP_DIR",
                "from config.settings import STT, TEMP_DIR\n"
                "try:\n"
                "    from config.settings import MICROPHONE_INDEX\n"
                "except ImportError:\n"
                "    MICROPHONE_INDEX = None"
            )
            content = content.replace(
                "with sr.Microphone() as source:",
                "mic_args = {'device_index': MICROPHONE_INDEX} if MICROPHONE_INDEX is not None else {}\n"
                "            with sr.Microphone(**mic_args) as source:"
            )
        with open(stt_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ {stt_path} 업데이트 완료")
        print(f"\n  이제 python main.py를 실행하면 장치 [{idx}]을 사용합니다.")
    else:
        main()