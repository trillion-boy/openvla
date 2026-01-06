"""
OpenVLA 전체 Pipeline 시뮬레이션 (모델 없이)

목표: 실제 모델 없이도 전체 흐름 체험
1. 이미지 + 명령어 입력
2. Action tokens 생성 (시뮬레이션)
3. Token → Normalized action
4. Un-normalization → 실제 로봇 명령

GPU 불필요, numpy만으로 실행 가능
"""

import numpy as np
import json
from pathlib import Path


# ============================================================
# SimpleActionTokenizer (이전 코드 재사용)
# ============================================================

class SimpleActionTokenizer:
    """기반: prismatic/vla/action_tokenizer.py"""

    def __init__(self, vocab_size: int = 32000, n_bins: int = 256):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Token IDs → 정규화 action"""
        discretized_actions = token_ids - (self.vocab_size - self.n_bins)
        discretized_actions = np.clip(discretized_actions, 0, self.n_bins - 1)
        return self.bin_centers[discretized_actions]


# ============================================================
# OpenVLA Pipeline 시뮬레이션
# ============================================================

class OpenVLASimulator:
    """
    실제 OpenVLA 모델의 동작을 시뮬레이션

    기반: prismatic/models/vlas/openvla.py:50-103
    """

    def __init__(self, dataset_statistics_path: str):
        # Dataset statistics 로드
        with open(dataset_statistics_path, 'r') as f:
            self.dataset_statistics = json.load(f)

        # Action tokenizer 초기화
        self.action_tokenizer = SimpleActionTokenizer(vocab_size=32000, n_bins=256)

        print("✅ OpenVLA Simulator 초기화 완료")
        print(f"   사용 가능한 데이터셋: {list(self.dataset_statistics.keys())}")

    def predict_action(self,
                      image_description: str,
                      instruction: str,
                      unnorm_key: str = "bridge_orig",
                      simulation_mode: str = "random") -> dict:
        """
        Action 예측 시뮬레이션

        Args:
            image_description: 이미지 설명 (실제 이미지 대신)
            instruction: 로봇 명령어
            unnorm_key: Dataset 이름
            simulation_mode: "random" 또는 "specific"

        Returns:
            dict: 전체 pipeline 결과
        """
        print("\n" + "=" * 70)
        print("🤖 OpenVLA Action Prediction")
        print("=" * 70)

        # Step 1: 입력 처리
        print(f"\n[입력]")
        print(f"  이미지: {image_description}")
        print(f"  명령어: {instruction}")
        print(f"  Dataset: {unnorm_key}")

        # Step 2: Action tokens 생성 (시뮬레이션)
        # 실제로는 vla.generate()가 실행됨
        print(f"\n[Step 1] 모델이 action tokens 생성 중... (시뮬레이션)")

        if simulation_mode == "random":
            # 랜덤 토큰 생성 (realistic range)
            action_token_ids = np.random.randint(31744, 32000, size=7)
        else:
            # 특정 토큰 (예: 약간의 전진 동작)
            action_token_ids = np.array([31872, 31800, 31850, 31872, 31872, 31872, 31950])

        print(f"  생성된 token IDs: {action_token_ids}")

        # Step 3: Token → Normalized action
        print(f"\n[Step 2] Token IDs → 정규화 action 변환")
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids)
        print(f"  정규화 actions ([-1, 1]): {normalized_actions}")

        # Step 4: Un-normalization
        print(f"\n[Step 3] Un-normalization (정규화 → 실제 로봇 명령)")
        action_stats = self.dataset_statistics[unnorm_key]["action"]
        q01 = np.array(action_stats["q01"])
        q99 = np.array(action_stats["q99"])
        mask = np.array(action_stats.get("mask", [True]*7))

        # openvla.py:97-101의 공식
        real_actions = 0.5 * (normalized_actions + 1) * (q99 - q01) + q01
        real_actions = np.where(mask, real_actions, normalized_actions)

        print(f"  실제 actions: {real_actions}")

        # Step 5: 해석
        print(f"\n[Step 4] 로봇 명령 해석")
        self._interpret_actions(real_actions, unnorm_key)

        # 결과 반환
        return {
            "action_token_ids": action_token_ids,
            "normalized_actions": normalized_actions,
            "real_actions": real_actions,
            "unnorm_key": unnorm_key
        }

    def _interpret_actions(self, actions: np.ndarray, dataset_name: str):
        """Action 값 해석"""
        dim_names = [
            "X-axis delta (m)",
            "Y-axis delta (m)",
            "Z-axis delta (m)",
            "Roll delta (rad)",
            "Pitch delta (rad)",
            "Yaw delta (rad)",
            "Gripper"
        ]

        for i, (name, value) in enumerate(zip(dim_names, actions)):
            if i < 6:
                print(f"  [{i}] {name:20s}: {value:+.6f}")
            else:
                status = "OPEN" if value > 0.5 else "CLOSE"
                print(f"  [{i}] {name:20s}: {value:.2f} ({status})")

        # 움직임 요약
        print(f"\n📊 움직임 요약:")
        if abs(actions[0]) > 0.001:
            direction = "앞으로" if actions[0] > 0 else "뒤로"
            print(f"  - X축: {direction} {abs(actions[0]*100):.2f}cm 이동")
        if abs(actions[1]) > 0.001:
            direction = "왼쪽으로" if actions[1] > 0 else "오른쪽으로"
            print(f"  - Y축: {direction} {abs(actions[1]*100):.2f}cm 이동")
        if abs(actions[2]) > 0.001:
            direction = "위로" if actions[2] > 0 else "아래로"
            print(f"  - Z축: {direction} {abs(actions[2]*100):.2f}cm 이동")

        gripper_action = "열기" if actions[6] > 0.5 else "닫기"
        print(f"  - Gripper: {gripper_action}")


# ============================================================
# 사용 예제
# ============================================================

def example_1_random_action():
    """예제 1: 랜덤 action 생성"""
    print("\n" + "=" * 70)
    print("예제 1: 랜덤 Action 생성")
    print("=" * 70)

    simulator = OpenVLASimulator("example_dataset_statistics.json")

    result = simulator.predict_action(
        image_description="책상 위에 파란 블록이 있음",
        instruction="pick up the blue block",
        unnorm_key="bridge_orig",
        simulation_mode="random"
    )

    return result


def example_2_specific_action():
    """예제 2: 특정 동작 시뮬레이션"""
    print("\n" + "=" * 70)
    print("예제 2: 특정 동작 (전진 + 그립)")
    print("=" * 70)

    simulator = OpenVLASimulator("example_dataset_statistics.json")

    result = simulator.predict_action(
        image_description="컵이 10cm 앞에 있음",
        instruction="grasp the cup",
        unnorm_key="bridge_orig",
        simulation_mode="specific"
    )

    return result


def example_3_multiple_predictions():
    """예제 3: 여러 번 예측 (trajectory)"""
    print("\n" + "=" * 70)
    print("예제 3: 연속 Action (Trajectory 시뮬레이션)")
    print("=" * 70)

    simulator = OpenVLASimulator("example_dataset_statistics.json")

    instructions = [
        "move forward to the object",
        "align with the object",
        "grasp the object",
        "lift the object"
    ]

    print("\n🎬 시나리오: 물체 집기")
    trajectory = []

    for i, instruction in enumerate(instructions):
        print(f"\n--- Step {i+1}: {instruction} ---")
        result = simulator.predict_action(
            image_description=f"현재 위치에서 물체까지 거리: {10-i*2}cm",
            instruction=instruction,
            unnorm_key="bridge_orig",
            simulation_mode="random"
        )
        trajectory.append(result)

    return trajectory


def compare_datasets():
    """예제 4: 다른 데이터셋 비교"""
    print("\n" + "=" * 70)
    print("예제 4: 데이터셋별 Un-normalization 비교")
    print("=" * 70)

    simulator = OpenVLASimulator("example_dataset_statistics.json")

    # 같은 정규화 action을 다른 dataset statistics로 변환
    normalized = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0])

    print(f"\n동일한 정규화 action: {normalized}")

    for dataset_name in ["bridge_orig", "fractal20220817_data"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        stats = simulator.dataset_statistics[dataset_name]["action"]
        q01 = np.array(stats["q01"])
        q99 = np.array(stats["q99"])

        real_actions = 0.5 * (normalized + 1) * (q99 - q01) + q01
        real_actions[6] = normalized[6]  # Gripper

        print(f"  실제 actions: {real_actions}")
        print(f"  X축 이동: {real_actions[0]*100:.2f}cm")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 OpenVLA Full Pipeline 시뮬레이션")
    print("=" * 70)
    print("\n이 코드는 실제 모델 없이 OpenVLA의 동작을 체험할 수 있습니다.")
    print("GPU 불필요, numpy만으로 실행 가능!")

    # 파일 확인
    if not Path("example_dataset_statistics.json").exists():
        print("\n❌ example_dataset_statistics.json 파일이 필요합니다!")
        exit(1)

    # 예제 실행
    print("\n" + "🎯 " * 20)
    example_1_random_action()

    print("\n" + "🎯 " * 20)
    example_2_specific_action()

    print("\n" + "🎯 " * 20)
    example_3_multiple_predictions()

    print("\n" + "🎯 " * 20)
    compare_datasets()

    print("\n" + "=" * 70)
    print("✅ 시뮬레이션 완료!")
    print("=" * 70)
    print("\n다음 단계:")
    print("  1. 이 코드로 action pipeline 이해")
    print("  2. 실제 모델 다운로드 (GPU 필요)")
    print("  3. 실제 이미지로 추론 실행")
