"""
OpenVLA Action Token 처리 실습

목표: OpenVLA의 action 처리 pipeline 이해
1. 모델이 생성한 7개 token IDs 추출
2. Token → 정규화된 action [-1, 1] 변환
3. Un-normalization → 실제 로봇 명령 변환

기반 코드:
- prismatic/models/vlas/openvla.py (predict_action 함수)
- prismatic/models/action_tokenizer.py (ActionTokenizer 클래스)
- prismatic/vla/datasets/rlds/utils/data_utils.py (정규화 관련)
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image


# ============================================================
# Step 1: ActionTokenizer 이해
# ============================================================

class SimpleActionTokenizer:
    """
    OpenVLA의 ActionTokenizer 간소화 버전

    기반: prismatic/models/action_tokenizer.py:40-88

    핵심 개념:
    - 연속 action 값 [-1, 1]을 256개의 bin으로 discretize
    - 각 bin은 하나의 token ID에 매핑
    - Vocabulary의 마지막 256개를 action tokens로 사용
    """

    def __init__(self, vocab_size: int = 32000, n_bins: int = 256):
        self.vocab_size = vocab_size
        self.n_bins = n_bins

        # Bin 경계: [-1, 1] 범위를 256개로 분할
        # 기반: action_tokenizer.py:48-49
        self.bins = np.linspace(-1, 1, n_bins + 1)

        # 각 bin의 중심값 (역변환 시 사용)
        # 기반: action_tokenizer.py:50
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        print(f"=== ActionTokenizer 초기화 ===")
        print(f"Vocab size: {vocab_size}")
        print(f"Number of bins: {n_bins}")
        print(f"Bin range: [{self.bins[0]}, {self.bins[-1]}]")
        print(f"Bin width: {self.bins[1] - self.bins[0]:.4f}")

    def encode(self, action: np.ndarray) -> np.ndarray:
        """
        연속 action → token IDs

        기반: action_tokenizer.py:68-73
        """
        # Clipping
        action = np.clip(action, -1.0, 1.0)

        # Discretization
        discretized = np.digitize(action, self.bins) - 1
        discretized = np.clip(discretized, 0, self.n_bins - 1)

        # Token ID 변환 (vocab의 마지막 256개 사용)
        token_ids = self.vocab_size - self.n_bins + discretized

        return token_ids

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Token IDs → 정규화된 연속 action

        기반: action_tokenizer.py:83-88

        Args:
            token_ids: [action_dim] shape의 token IDs

        Returns:
            actions: [action_dim] shape의 정규화 action [-1, 1]
        """
        # Token ID를 bin index로 역변환
        discretized_actions = token_ids - (self.vocab_size - self.n_bins)

        # Safety clipping
        discretized_actions = np.clip(discretized_actions, 0, self.n_bins - 1)

        # Bin index → 연속 값 (bin 중심값 사용)
        continuous_actions = self.bin_centers[discretized_actions]

        return continuous_actions


# ============================================================
# Step 2: Un-normalization 이해
# ============================================================

def unnormalize_actions(normalized_actions: np.ndarray,
                       action_stats: dict) -> np.ndarray:
    """
    정규화된 action [-1, 1] → 실제 로봇 명령

    기반: prismatic/models/vlas/openvla.py:94-101

    OpenVLA는 BOUNDS_Q99 정규화 사용:
    - Forward:  norm = 2 * (action - q01) / (q99 - q01) - 1
    - Backward: action = 0.5 * (norm + 1) * (q99 - q01) + q01

    참고: prismatic/vla/datasets/rlds/utils/data_utils.py:49-54
          NormalizationType.BOUNDS_Q99
    """
    action_low = np.array(action_stats["q01"])
    action_high = np.array(action_stats["q99"])

    # Un-normalization 공식 (openvla.py:97-101)
    real_actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low

    # Mask 적용 (일부 차원은 정규화 안 함)
    # 예: Gripper는 이미 [0, 1] 범위라 정규화 안 함
    mask = action_stats.get("mask", np.ones_like(action_low, dtype=bool))
    real_actions = np.where(mask, real_actions, normalized_actions)

    return real_actions


# ============================================================
# Step 3: 전체 Pipeline 예제
# ============================================================

def demonstrate_action_pipeline():
    """
    OpenVLA의 action 처리 전체 흐름 시연

    기반: prismatic/models/vlas/openvla.py:84-103
    """
    print("\n" + "=" * 60)
    print("OpenVLA Action Processing Pipeline")
    print("=" * 60)

    # 예제 데이터
    vocab_size = 32000
    action_dim = 7  # Bridge dataset: [6 EEF + 1 gripper]

    # Step 1: 모델이 생성한 token IDs (예시)
    # 실제로는 vla.generate()의 결과
    print("\n[Step 1] 모델 생성 결과")
    generated_ids = np.array([31800, 31850, 31900, 31750, 31780, 31820, 31950])
    print(f"Generated token IDs (마지막 7개): {generated_ids}")
    print(f"  - 범위: [{generated_ids.min()}, {generated_ids.max()}]")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Action token 범위: [{vocab_size - 256}, {vocab_size}]")

    # Step 2: Token → 정규화 action
    print("\n[Step 2] Token IDs → 정규화 action")
    tokenizer = SimpleActionTokenizer(vocab_size=vocab_size)
    normalized_actions = tokenizer.decode_token_ids_to_actions(generated_ids)
    print(f"정규화 actions: {normalized_actions}")
    print(f"  - 범위: [{normalized_actions.min():.3f}, {normalized_actions.max():.3f}]")

    # Step 3: Dataset statistics 로드 (예시)
    print("\n[Step 3] Dataset statistics")
    # 실제로는 dataset_statistics.json에서 로드
    # 기반: prismatic/vla/datasets/rlds/utils/data_utils.py:185-293
    bridge_stats = {
        "q01": np.array([-0.4, -0.35, -0.5, -0.3, -0.25, -0.3, 0.0]),
        "q99": np.array([0.45, 0.38, 0.52, 0.32, 0.28, 0.32, 1.0]),
        "mask": np.array([True, True, True, True, True, True, False])
    }
    print(f"q01 (1% quantile):  {bridge_stats['q01']}")
    print(f"q99 (99% quantile): {bridge_stats['q99']}")
    print(f"Action range:       {bridge_stats['q99'] - bridge_stats['q01']}")

    # Step 4: Un-normalization
    print("\n[Step 4] Un-normalization → 실제 로봇 명령")
    real_actions = unnormalize_actions(normalized_actions, bridge_stats)
    print(f"실제 actions: {real_actions}")
    print(f"\n차원별 해석 (Bridge dataset):")
    print(f"  [0] X-axis delta (m):        {real_actions[0]:+.4f}")
    print(f"  [1] Y-axis delta (m):        {real_actions[1]:+.4f}")
    print(f"  [2] Z-axis delta (m):        {real_actions[2]:+.4f}")
    print(f"  [3] Roll delta (rad):        {real_actions[3]:+.4f}")
    print(f"  [4] Pitch delta (rad):       {real_actions[4]:+.4f}")
    print(f"  [5] Yaw delta (rad):         {real_actions[5]:+.4f}")
    print(f"  [6] Gripper (0=close, 1=open): {real_actions[6]:.2f}")


# ============================================================
# Step 4: 역변환 검증
# ============================================================

def verify_normalization_inverse():
    """
    정규화 ↔ 역정규화가 올바른 역함수인지 검증

    기반: data_utils.py의 정규화 공식
    """
    print("\n" + "=" * 60)
    print("정규화/역정규화 검증")
    print("=" * 60)

    # 예제 statistics
    q01, q99 = -0.4, 0.45

    # 원본 action
    original_action = 0.10  # 10cm
    print(f"\n원본 action: {original_action:.3f} m")

    # Forward: 정규화
    normalized = 2 * (original_action - q01) / (q99 - q01) - 1
    print(f"정규화 후:   {normalized:.3f}")

    # Backward: 역정규화
    recovered = 0.5 * (normalized + 1) * (q99 - q01) + q01
    print(f"복원 후:     {recovered:.3f} m")

    # 검증
    error = abs(original_action - recovered)
    print(f"복원 오차:   {error:.6f}")
    assert error < 1e-6, "정규화/역정규화 오차 발생!"
    print("✅ 검증 통과!")


# ============================================================
# Step 5: 실제 코드 위치 참고
# ============================================================

def print_code_references():
    """
    OpenVLA 코드베이스의 관련 파일들
    """
    print("\n" + "=" * 60)
    print("OpenVLA 코드 참고")
    print("=" * 60)

    references = [
        {
            "파일": "prismatic/models/vlas/openvla.py",
            "함수/클래스": "OpenVLA.predict_action()",
            "라인": "84-103",
            "설명": "전체 action 생성 pipeline"
        },
        {
            "파일": "prismatic/models/action_tokenizer.py",
            "함수/클래스": "ActionTokenizer",
            "라인": "40-88",
            "설명": "Token ↔ Action 변환"
        },
        {
            "파일": "prismatic/vla/datasets/rlds/utils/data_utils.py",
            "함수/클래스": "get_dataset_statistics()",
            "라인": "185-293",
            "설명": "Dataset statistics 계산/로드"
        },
        {
            "파일": "prismatic/vla/datasets/rlds/utils/data_utils.py",
            "함수/클래스": "NormalizationType.BOUNDS_Q99",
            "라인": "49-54",
            "설명": "정규화 방식 정의"
        },
        {
            "파일": "prismatic/vla/datasets/rlds/oxe/configs.py",
            "함수/클래스": "OXE_DATASET_CONFIGS['bridge_orig']",
            "라인": "79-85",
            "설명": "Bridge dataset 설정"
        },
    ]

    for ref in references:
        print(f"\n📄 {ref['파일']}:{ref['라인']}")
        print(f"   {ref['함수/클래스']}")
        print(f"   → {ref['설명']}")


# ============================================================
# 연습 문제
# ============================================================

def exercise_1_token_conversion():
    """
    연습 1: Token ID ↔ Action 변환

    문제: 다음 token ID가 어떤 action 값으로 변환되는지 계산하세요.
    """
    print("\n" + "=" * 60)
    print("연습 문제 1: Token → Action 변환")
    print("=" * 60)

    vocab_size = 32000
    n_bins = 256

    # Token ID 예시
    token_id = 31872  # vocab_size - 128

    print(f"\n주어진 값:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Bins: {n_bins}")
    print(f"  Token ID: {token_id}")

    # TODO: 여기서부터 계산
    # 힌트 1: bin_index = token_id - (vocab_size - n_bins)
    # 힌트 2: bin_centers = linspace(-1, 1, n_bins)의 중심값들

    bin_index = token_id - (vocab_size - n_bins)
    bins = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    action_value = bin_centers[bin_index]

    print(f"\n정답:")
    print(f"  Bin index: {bin_index}")
    print(f"  Action value: {action_value:.4f}")


def exercise_2_unnormalization():
    """
    연습 2: Un-normalization 계산

    문제: 정규화된 action을 실제 로봇 명령으로 변환하세요.
    """
    print("\n" + "=" * 60)
    print("연습 문제 2: Un-normalization")
    print("=" * 60)

    # 주어진 값
    normalized_actions = np.array([0.5, -0.3, 0.8, 0.0, -1.0, 1.0, 0.9])
    q01 = np.array([-0.4, -0.35, -0.5, -0.3, -0.25, -0.3, 0.0])
    q99 = np.array([0.45, 0.38, 0.52, 0.32, 0.28, 0.32, 1.0])

    print(f"\n주어진 값:")
    print(f"  정규화 actions: {normalized_actions}")
    print(f"  q01: {q01}")
    print(f"  q99: {q99}")

    # TODO: Un-normalization 수식 적용
    # 공식: action = 0.5 * (norm + 1) * (q99 - q01) + q01

    real_actions = 0.5 * (normalized_actions + 1) * (q99 - q01) + q01

    print(f"\n정답:")
    print(f"  실제 actions: {real_actions}")
    print(f"\n검증 (차원별):")
    for i in range(7):
        print(f"    Dim {i}: norm={normalized_actions[i]:+.2f} → real={real_actions[i]:+.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OpenVLA Action Token 처리 실습")
    print("=" * 60)

    # 전체 pipeline 시연
    demonstrate_action_pipeline()

    # 정규화 검증
    verify_normalization_inverse()

    # 코드 참고 정보
    print_code_references()

    # 연습 문제
    exercise_1_token_conversion()
    exercise_2_unnormalization()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("  1. OpenVLA 실제 코드 읽기 (위 참고 파일들)")
    print("  2. Dataset statistics 파일 찾기")
    print("  3. 실제 모델로 추론 실행해보기")
