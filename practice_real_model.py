"""
실제 OpenVLA 모델로 Action Tokens 얻기

요구사항:
- GPU (CUDA)
- 최소 16GB VRAM
- transformers, torch, PIL 라이브러리

실행 전:
pip install transformers torch pillow huggingface_hub
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path


def load_openvla_model(model_id: str = "openvla/openvla-7b"):
    """
    OpenVLA 모델 로드

    기반: prismatic/models/load.py
    """
    print("=" * 70)
    print("🤖 OpenVLA 모델 로드")
    print("=" * 70)

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        print("❌ transformers가 설치되지 않았습니다!")
        print("   pip install transformers torch pillow")
        return None, None

    print(f"\n모델 다운로드 중: {model_id}")
    print("(첫 실행 시 ~14GB 다운로드, 시간이 걸릴 수 있습니다)")

    # Processor 로드
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\n⚠️  GPU를 찾을 수 없습니다. CPU로 실행하면 매우 느립니다!")

    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    print(f"\n✅ 모델 로드 완료! (Device: {device})")

    return vla, processor


def predict_action_with_tokens(
    vla,
    processor,
    image_path: str,
    instruction: str,
    unnorm_key: str = "bridge_orig"
):
    """
    실제 OpenVLA 모델로 action 예측 + token 추출

    기반: prismatic/models/vlas/openvla.py:50-103
    """
    print("\n" + "=" * 70)
    print("🎯 Action Prediction")
    print("=" * 70)

    # 입력 준비
    print(f"\n[입력]")
    print(f"  이미지: {image_path}")
    print(f"  명령어: {instruction}")

    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    # Processor로 입력 처리
    inputs = processor(prompt, image).to(vla.device, dtype=vla.dtype)

    # Action 예측 (predict_action 메서드 사용)
    print(f"\n[Step 1] 모델 추론 중...")
    with torch.no_grad():
        # 방법 1: predict_action 사용 (un-normalization 포함)
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    print(f"  ✅ 예측 완료!")
    print(f"  실제 로봇 명령: {action}")

    # 방법 2: 수동으로 tokens 추출
    print(f"\n[Step 2] Token IDs 수동 추출...")
    with torch.no_grad():
        # Generate로 직접 생성
        action_dim = 7
        generated_ids = vla.generate(
            **inputs,
            max_new_tokens=action_dim,
            do_sample=False
        )

        # 마지막 7개 token 추출 (openvla.py:90)
        action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()

    print(f"  생성된 token IDs: {action_token_ids}")

    # Token → Normalized action
    normalized_actions = vla.action_tokenizer.decode_token_ids_to_actions(action_token_ids)
    print(f"  정규화 actions: {normalized_actions}")

    # Un-normalization 수동 수행
    action_stats = vla.get_action_stats(unnorm_key)
    q01 = np.array(action_stats["q01"])
    q99 = np.array(action_stats["q99"])
    mask = action_stats.get("mask", np.ones_like(q01, dtype=bool))

    manual_action = 0.5 * (normalized_actions + 1) * (q99 - q01) + q01
    manual_action = np.where(mask, manual_action, normalized_actions)

    print(f"  수동 un-normalization: {manual_action}")

    # 검증
    print(f"\n[검증]")
    print(f"  predict_action() 결과: {action}")
    print(f"  수동 계산 결과:        {manual_action}")
    print(f"  차이: {np.abs(action - manual_action)}")

    return {
        "action": action,
        "action_token_ids": action_token_ids,
        "normalized_actions": normalized_actions,
        "manual_action": manual_action
    }


def interactive_demo(vla, processor):
    """대화형 데모"""
    print("\n" + "=" * 70)
    print("🎮 대화형 Action 예측")
    print("=" * 70)

    print("\n사용법:")
    print("  1. 이미지 경로 입력")
    print("  2. 로봇 명령어 입력")
    print("  3. Action tokens 확인!")
    print("  (종료: Ctrl+C)")

    try:
        while True:
            print("\n" + "-" * 70)
            image_path = input("이미지 경로: ").strip()
            if not image_path or not Path(image_path).exists():
                print("❌ 유효한 이미지 경로를 입력하세요")
                continue

            instruction = input("로봇 명령어: ").strip()
            if not instruction:
                print("❌ 명령어를 입력하세요")
                continue

            result = predict_action_with_tokens(
                vla, processor, image_path, instruction
            )

            print("\n✅ 예측 완료!")

    except KeyboardInterrupt:
        print("\n\n종료합니다.")


# ============================================================
# 예제 실행
# ============================================================

def example_with_dummy_image():
    """더미 이미지로 테스트"""
    print("\n" + "=" * 70)
    print("예제: 더미 이미지로 테스트")
    print("=" * 70)

    # 더미 이미지 생성
    dummy_image = Image.new('RGB', (224, 224), color='blue')
    dummy_path = "/tmp/dummy_image.png"
    dummy_image.save(dummy_path)

    print(f"\n더미 이미지 생성: {dummy_path}")

    # 모델 로드
    vla, processor = load_openvla_model()

    if vla is None:
        print("\n❌ 모델 로드 실패")
        return

    # 예측
    result = predict_action_with_tokens(
        vla,
        processor,
        dummy_path,
        "pick up the blue block",
        unnorm_key="bridge_orig"
    )

    # Token 분석
    print("\n" + "=" * 70)
    print("📊 Token 분석")
    print("=" * 70)

    token_ids = result["action_token_ids"]
    vocab_size = 32000

    print(f"\nToken IDs:")
    for i, tid in enumerate(token_ids):
        bin_idx = tid - (vocab_size - 256)
        norm_val = result["normalized_actions"][i]
        real_val = result["action"][i]
        print(f"  Dim {i}: token_id={tid:5d} → bin={bin_idx:3d} → norm={norm_val:+.3f} → real={real_val:+.6f}")

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("🤖 실제 OpenVLA 모델로 Action Tokens 얻기")
    print("=" * 70)

    print("\n선택하세요:")
    print("  1. 더미 이미지로 테스트")
    print("  2. 대화형 모드")
    print("  3. 종료")

    try:
        choice = input("\n선택 (1/2/3): ").strip()

        if choice == "1":
            example_with_dummy_image()

        elif choice == "2":
            vla, processor = load_openvla_model()
            if vla is not None:
                interactive_demo(vla, processor)

        elif choice == "3":
            print("종료합니다.")
            sys.exit(0)

        else:
            print("❌ 잘못된 선택")

    except KeyboardInterrupt:
        print("\n\n종료합니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
