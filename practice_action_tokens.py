"""

Practice: 7 Action Tokens 추출 및 Un-normalization

 

목표: OpenVLA의 action 처리 flow를 이해

1. 모델이 생성한 token IDs 추출

2. Token → 정규화된 action [-1, 1] 변환

3. Un-normalization → 실제 로봇 명령 변환

"""

 

import numpy as np

import torch

from pathlib import Path

from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor

 

# ============================================================

# Step 1: 모델 로드 및 추론

# ============================================================

 

def load_model():

    """OpenVLA 모델과 processor 로드"""

    model_path = "openvla/openvla-7b"  # 또는 local path

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(

        model_path,

        torch_dtype=torch.bfloat16,

        trust_remote_code=True,

    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return vla, processor

 

 

def predict_and_extract_tokens(vla, processor, image_path: str, instruction: str):

    """

    이미지와 명령어를 받아서 7개의 action token을 추출합니다.

 

    Returns:

        generated_ids: 모델이 생성한 전체 token sequence

        action_token_ids: 마지막 7개 action token IDs

    """

    # 입력 준비

    image = Image.open(image_path).convert("RGB")

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

 

    inputs = processor(prompt, image).to(vla.device, dtype=vla.dtype)

 

    # 생성 (greedy decoding)

    action_dim = 7  # Bridge dataset의 action 차원

    generated_ids = vla.generate(

        **inputs,

        max_new_tokens=action_dim,  # 7개 토큰만 생성

        do_sample=False,            # deterministic

    )

 

    # 마지막 7개 토큰 추출

    action_token_ids = generated_ids[0, -action_dim:]

 

    print(f"생성된 전체 sequence 길이: {generated_ids.shape[1]}")

    print(f"추출된 action token IDs: {action_token_ids.cpu().numpy()}")

 

    return generated_ids, action_token_ids

 

 

# ============================================================

# Step 2: Token → 정규화 Action 변환

# ============================================================

 

class SimpleActionTokenizer:

    """

    OpenVLA의 ActionTokenizer를 간소화한 버전

 

    핵심 개념:

    - 연속 action 값 [-1, 1]을 256개의 bin으로 discretize

    - 각 bin은 하나의 token ID에 매핑됨

    - Vocabulary의 마지막 256개를 action tokens로 사용

    """

    def __init__(self, vocab_size: int, n_bins: int = 256):

        self.vocab_size = vocab_size

        self.n_bins = n_bins

 

        # Bin 경계 생성: [-1, 1] 범위를 256개로 나눔

        self.bins = np.linspace(-1, 1, n_bins + 1)

 

        # 각 bin의 중심값 (un-discretize 시 사용)

        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

 

        print(f"Bin 예시: {self.bins[:5]} ... {self.bins[-5:]}")

        print(f"Bin 중심 예시: {self.bin_centers[:5]} ... {self.bin_centers[-5:]}")

 

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:

        """

        Token IDs → 정규화된 연속 action 변환

 

        Args:

            token_ids: [action_dim] shape의 token IDs

 

        Returns:

            actions: [action_dim] shape의 정규화 action [-1, 1]

        """

        # Token ID를 bin index로 변환

        # (vocab_size - 256) ~ vocab_size 범위를 0~255로 매핑

        discretized_actions = self.vocab_size - token_ids

 

        # Clipping (안전장치)

        discretized_actions = np.clip(discretized_actions - 1, 0, self.n_bins - 1)

 

        # Bin index → 연속 값 (bin의 중심값 사용)

        continuous_actions = self.bin_centers[discretized_actions]

 

        print(f"\nToken IDs: {token_ids}")

        print(f"Bin indices: {discretized_actions}")

        print(f"정규화 actions: {continuous_actions}")

 

        return continuous_actions

 

 

# ============================================================

# Step 3: Un-normalization (핵심!)

# ============================================================

 

def load_dataset_statistics(stats_path: str, dataset_key: str = "bridge_orig"):

    """

    Dataset statistics JSON 파일 로드

 

    이 파일에는 각 데이터셋의 action 통계가 저장되어 있습니다:

    - q01, q99: 1%, 99% quantile (outlier 제거)

    - mean, std: 평균, 표준편차

 

    OpenVLA는 BOUNDS_Q99 정규화 방식을 사용:

    normalized = 2 * (action - q01) / (q99 - q01) - 1  →  [-1, 1]

    """

    import json

 

    with open(stats_path, 'r') as f:

        all_stats = json.load(f)

 

    if dataset_key not in all_stats:

        raise ValueError(f"Dataset '{dataset_key}' not found in statistics file")

 

    stats = all_stats[dataset_key]["action"]

 

    print(f"\n=== {dataset_key} Dataset Statistics ===")

    print(f"q01 (1% quantile):  {np.array(stats['q01'])}")

    print(f"q99 (99% quantile): {np.array(stats['q99'])}")

    print(f"Range per dim: {np.array(stats['q99']) - np.array(stats['q01'])}")

 

    return stats

 

 

def unnormalize_actions(normalized_actions: np.ndarray, action_stats: dict) -> np.ndarray:

    """

    정규화 [-1, 1] → 실제 로봇 명령 변환

 

    역변환 공식 (BOUNDS_Q99):

    action = (normalized + 1) / 2 * (q99 - q01) + q01

           = 0.5 * (normalized + 1) * (q99 - q01) + q01

 

    예시:

    - normalized = -1 → action = q01 (최소값)

    - normalized =  0 → action = (q01 + q99) / 2 (중앙값)

    - normalized = +1 → action = q99 (최대값)

    """

    action_low = np.array(action_stats["q01"])

    action_high = np.array(action_stats["q99"])

 

    # Un-normalization

    real_actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low

 

    print(f"\n=== Un-normalization ===")

    print(f"정규화 actions: {normalized_actions}")

    print(f"실제 actions:   {real_actions}")

    print(f"\n차원별 해석 (Bridge dataset 기준):")

    print(f"  [0-2] EEF XYZ 델타 (m):     {real_actions[:3]}")

    print(f"  [3-5] Roll-Pitch-Yaw (rad): {real_actions[3:6]}")

    print(f"  [6]   Gripper (0=close, 1=open): {real_actions[6]}")

 

    return real_actions

 

 

# ============================================================

# Step 4: 전체 파이프라인 실행

# ============================================================

 

def main():

    """

    전체 flow 실습:

    모델 추론 → Token 추출 → 정규화 action → Un-normalization

    """

    # 설정

    image_path = "path/to/your/image.png"  # 실제 이미지 경로로 변경

    instruction = "pick up the blue block"

    stats_path = "path/to/dataset_statistics.json"  # 실제 경로로 변경

 

    print("=" * 60)

    print("OpenVLA Action Processing Pipeline 실습")

    print("=" * 60)

 

    # Step 1: 모델 로드 및 추론

    print("\n[Step 1] 모델 로드 및 action token 생성...")

    vla, processor = load_model()

    generated_ids, action_token_ids = predict_and_extract_tokens(

        vla, processor, image_path, instruction

    )

 

    # Step 2: Token → 정규화 action

    print("\n[Step 2] Token IDs → 정규화 action 변환...")

    tokenizer = SimpleActionTokenizer(vocab_size=vla.config.vocab_size)

    normalized_actions = tokenizer.decode_token_ids_to_actions(

        action_token_ids.cpu().numpy()

    )

 

    # Step 3: Un-normalization

    print("\n[Step 3] Dataset statistics 로드...")

    action_stats = load_dataset_statistics(stats_path, dataset_key="bridge_orig")

 

    print("\n[Step 4] Un-normalization 수행...")

    real_actions = unnormalize_actions(normalized_actions, action_stats)

 

    print("\n" + "=" * 60)

    print("✅ 완료! 실제 로봇에 전달할 action:")

    print(f"   {real_actions}")

    print("=" * 60)

 

    return real_actions

 

 

# ============================================================

# 연습 문제

# ============================================================

 

def exercise_1():

    """

    연습 1: 정규화/역정규화 수식 이해

 

    문제: 다음 정규화 action이 실제 로봇 명령으로 어떻게 변환되는지 계산하세요.

 

    주어진 값:

    - normalized_action = [0.5, -0.3, 0.8, 0.0, -1.0, 1.0, 0.9]

    - q01 = [-0.4, -0.35, -0.5, -0.3, -0.25, -0.3, 0.0]

    - q99 = [0.45, 0.38, 0.52, 0.32, 0.28, 0.32, 1.0]

 

    힌트: action = 0.5 * (norm + 1) * (q99 - q01) + q01

    """

    normalized = np.array([0.5, -0.3, 0.8, 0.0, -1.0, 1.0, 0.9])

    q01 = np.array([-0.4, -0.35, -0.5, -0.3, -0.25, -0.3, 0.0])

    q99 = np.array([0.45, 0.38, 0.52, 0.32, 0.28, 0.32, 1.0])

 

    # TODO: 여기에 코드를 작성하세요

    real_actions = None  # 계산 결과

 

    print(f"정답: {real_actions}")

 

 

def exercise_2():

    """

    연습 2: TACO 적용 시 주의사항

 

    문제: TACO로 모델의 logits를 제어할 때, 왜 un-normalization을 고려해야 할까요?

 

    시나리오:

    - 로봇을 X축으로 +10cm 이동시키고 싶음

    - 하지만 모델은 정규화 공간 [-1, 1]에서 생각함

    - q01[0] = -0.4, q99[0] = 0.45 (실제 범위 -40cm ~ +45cm)

 

    질문:

    1. +10cm 이동 = 정규화 공간의 몇인가?

    2. TACO 제약을 normalized 공간에 걸어야 하는가, real 공간에 걸어야 하는가?

    """

    target_movement_cm = 10.0

    q01_x = -40.0  # cm

    q99_x = 45.0   # cm

 

    # TODO: 정규화 공간의 값 계산

    # 역공식: normalized = 2 * (action - q01) / (q99 - q01) - 1

    normalized_target = None

 

    print(f"10cm 이동 = 정규화 공간의 {normalized_target}")

    print(f"→ TACO 제약은 {normalized_target} 근처 logits를 강화해야 함!")

 

 

if __name__ == "__main__":

    # main()  # 주석 해제하여 실행

 

    # 연습 문제 풀어보기

    print("\n" + "="*60)

    print("연습 문제 1: 정규화/역정규화 계산")

    print("="*60)

    exercise_1()

 

    print("\n" + "="*60)

    print("연습 문제 2: TACO 적용 시 주의사항")

    print("="*60)

    exercise_2()
