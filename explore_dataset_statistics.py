"""
Dataset Statistics 탐색
목표: OpenVLA의 dataset_statistics.json 파일 이해
- 파일 구조
- 각 필드의 의미
- Un-normalization에 사용되는 방법
기반: prismatic/vla/datasets/rlds/utils/data_utils.py:185-293
"""

import json
import numpy as np
from pathlib import Path


def load_and_explore_statistics(stats_file: str = "example_dataset_statistics.json"):
    """
    Dataset statistics 파일 로드 및 탐색
    """
    print("=" * 70)
    print("Dataset Statistics 탐색")
    print("=" * 70)

    # 파일 로드
    with open(stats_file, 'r') as f:
        all_stats = json.load(f)

    print(f"\n📄 파일: {stats_file}")
    print(f"포함된 데이터셋: {list(all_stats.keys())}")

    # 각 데이터셋별로 탐색
    for dataset_name, dataset_stats in all_stats.items():
        explore_dataset(dataset_name, dataset_stats)


def explore_dataset(dataset_name: str, dataset_stats: dict):
    """
    개별 데이터셋 통계 탐색
    """
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    # Action statistics
    if "action" in dataset_stats:
        print("\n[Action Statistics]")
        action_stats = dataset_stats["action"]

        print("\n필드 설명:")
        print("  - mean:   각 차원의 평균값")
        print("  - std:    각 차원의 표준편차")
        print("  - min:    최솟값 (절대 최솟값)")
        print("  - max:    최댓값 (절대 최댓값)")
        print("  - q01:    1% quantile (outlier 제거용)")
        print("  - q99:    99% quantile (outlier 제거용)")
        print("  - mask:   정규화 적용 여부 (True=정규화, False=그대로)")

        # 배열로 변환
        q01 = np.array(action_stats["q01"])
        q99 = np.array(action_stats["q99"])
        mean = np.array(action_stats["mean"])
        std = np.array(action_stats["std"])
        mask = np.array(action_stats["mask"])

        print(f"\nAction 차원 수: {len(q01)}")
        print(f"\n통계값 (7개 차원):")
        print(f"  q01: {q01}")
        print(f"  q99: {q99}")
        print(f"  mean: {mean}")
        print(f"  std: {std}")
        print(f"  mask: {mask}")

        # Action 범위 분석
        print(f"\n각 차원의 유효 범위 (q99 - q01):")
        action_range = q99 - q01
        for i, r in enumerate(action_range):
            dim_name = get_dimension_name(i, dataset_name)
            print(f"  [{i}] {dim_name:20s}: {r:.6f}")

        # Mask 설명
        print(f"\n정규화 마스크:")
        for i, m in enumerate(mask):
            dim_name = get_dimension_name(i, dataset_name)
            status = "정규화함" if m else "그대로"
            print(f"  [{i}] {dim_name:20s}: {status}")

        # 왜 gripper는 정규화 안 하나?
        if not mask[-1]:
            print(f"\n💡 Gripper(차원 6)를 정규화 안 하는 이유:")
            print(f"   - Gripper는 이미 [0, 1] 또는 {{-1, 1}} 범위로 표준화되어 있음")
            print(f"   - 0 = 닫힘(close), 1 = 열림(open)")
            print(f"   - 추가 정규화 불필요!")

    # Dataset 메타데이터
    if "num_transitions" in dataset_stats:
        print(f"\n[Dataset 메타데이터]")
        print(f"  총 transitions: {dataset_stats['num_transitions']:,}")
        print(f"  총 trajectories: {dataset_stats['num_trajectories']:,}")
        avg_steps = dataset_stats['num_transitions'] / dataset_stats['num_trajectories']
        print(f"  평균 trajectory 길이: {avg_steps:.1f} steps")


def get_dimension_name(dim: int, dataset_name: str) -> str:
    """
    차원 번호 → 의미 있는 이름 변환
    """
    # Bridge dataset의 경우
    if "bridge" in dataset_name.lower():
        names = [
            "X-axis delta (m)",
            "Y-axis delta (m)",
            "Z-axis delta (m)",
            "Roll delta (rad)",
            "Pitch delta (rad)",
            "Yaw delta (rad)",
            "Gripper (0/1)"
        ]
    else:
        names = [f"Dim {dim}" for dim in range(7)]

    return names[dim] if dim < len(names) else f"Dim {dim}"


def demonstrate_unnormalization():
    """
    Dataset statistics를 사용한 un-normalization 예제
    """
    print("\n" + "=" * 70)
    print("Un-normalization 실습")
    print("=" * 70)

    # Statistics 로드
    with open("example_dataset_statistics.json", 'r') as f:
        stats = json.load(f)

    bridge_stats = stats["bridge_orig"]["action"]
    q01 = np.array(bridge_stats["q01"])
    q99 = np.array(bridge_stats["q99"])
    mask = np.array(bridge_stats["mask"])

    # 정규화된 action (예시)
    normalized_actions = np.array([0.5, -0.3, 0.8, 0.0, -1.0, 1.0, 0.9])

    print(f"\n입력 (정규화된 actions): {normalized_actions}")
    print(f"범위: [-1, 1]")

    # Un-normalization (openvla.py:97-101 방식)
    real_actions = 0.5 * (normalized_actions + 1) * (q99 - q01) + q01

    # Mask 적용 (gripper는 그대로)
    real_actions = np.where(mask, real_actions, normalized_actions)

    print(f"\n출력 (실제 로봇 명령): {real_actions}")
    print(f"\n차원별 해석:")
    for i in range(7):
        dim_name = get_dimension_name(i, "bridge_orig")
        print(f"  [{i}] {dim_name:20s}: {real_actions[i]:+.6f}")

    # 검증: 범위 확인
    print(f"\n✅ 검증:")
    for i in range(6):  # Gripper 제외
        in_range = q01[i] <= real_actions[i] <= q99[i]
        status = "✓" if in_range else "✗"
        print(f"  [{i}] {status} q01 <= action <= q99: {in_range}")


def compare_normalization_methods():
    """
    다른 정규화 방법 비교
    """
    print("\n" + "=" * 70)
    print("정규화 방법 비교")
    print("=" * 70)

    # 예제 값
    action = 0.025  # 2.5cm
    q01, q99 = -0.033, 0.032
    mean, std = -0.0005, 0.013

    print(f"\n원본 action: {action:.4f} m (2.5cm)")
    print(f"q01={q01:.4f}, q99={q99:.4f}")
    print(f"mean={mean:.4f}, std={std:.4f}")

    # Method 1: BOUNDS (Min-Max)
    normalized_bounds = 2 * (action - q01) / (q99 - q01) - 1
    print(f"\n[1] BOUNDS (Min-Max):")
    print(f"    norm = 2 * (x - min) / (max - min) - 1")
    print(f"    결과: {normalized_bounds:.4f}")

    # Method 2: BOUNDS_Q99 (OpenVLA 사용)
    normalized_q99 = 2 * (action - q01) / (q99 - q01) - 1
    print(f"\n[2] BOUNDS_Q99 (OpenVLA) ⭐:")
    print(f"    norm = 2 * (x - q01) / (q99 - q01) - 1")
    print(f"    결과: {normalized_q99:.4f}")
    print(f"    특징: Outlier 제거, robust")

    # Method 3: NORMAL (Z-score)
    normalized_zscore = (action - mean) / std
    print(f"\n[3] NORMAL (Z-score):")
    print(f"    norm = (x - mean) / std")
    print(f"    결과: {normalized_zscore:.4f}")
    print(f"    특징: 범위가 [-1, 1]로 제한 안 됨!")

    print(f"\n💡 OpenVLA가 BOUNDS_Q99를 쓰는 이유:")
    print(f"   1. Outlier에 robust")
    print(f"   2. 항상 [-1, 1] 범위 (tokenization에 유리)")
    print(f"   3. 대부분의 데이터는 [-1, 1] 안에 들어옴")


def practical_exercise():
    """
    실전 연습: 특정 로봇 명령을 정규화 공간으로 변환
    """
    print("\n" + "=" * 70)
    print("실전 연습: 로봇 명령 → 정규화 값")
    print("=" * 70)

    # Statistics 로드
    with open("example_dataset_statistics.json", 'r') as f:
        stats = json.load(f)

    bridge_stats = stats["bridge_orig"]["action"]
    q01 = np.array(bridge_stats["q01"])
    q99 = np.array(bridge_stats["q99"])

    print("\n문제: 로봇에게 다음 명령을 내리고 싶습니다:")
    print("  - X축으로 +2cm 이동")
    print("  - Y축으로 -1cm 이동")
    print("  - Z축 변화 없음")
    print("  - 회전 없음")
    print("  - Gripper 열기")

    # 실제 명령 (meters)
    real_command = np.array([
        0.02,   # X: +2cm
        -0.01,  # Y: -1cm
        0.0,    # Z: 0
        0.0,    # Roll: 0
        0.0,    # Pitch: 0
        0.0,    # Yaw: 0
        1.0     # Gripper: open
    ])

    print(f"\n실제 명령 (meters/rad): {real_command}")

    # Forward: 정규화
    normalized_command = 2 * (real_command - q01) / (q99 - q01) - 1

    # Gripper는 그대로
    normalized_command[6] = real_command[6]

    print(f"\n정규화된 명령: {normalized_command}")
    print(f"\n각 차원:")
    for i in range(7):
        dim_name = get_dimension_name(i, "bridge_orig")
        print(f"  [{i}] {dim_name:20s}: {real_command[i]:+.4f} → {normalized_command[i]:+.4f}")

    # 검증: 역변환
    recovered = 0.5 * (normalized_command[:6] + 1) * (q99[:6] - q01[:6]) + q01[:6]
    recovered = np.append(recovered, normalized_command[6])

    print(f"\n✅ 역변환 검증:")
    print(f"  원본: {real_command}")
    print(f"  복원: {recovered}")
    print(f"  오차: {np.abs(real_command - recovered)}")


if __name__ == "__main__":
    # 파일 존재 확인
    if not Path("example_dataset_statistics.json").exists():
        print("❌ example_dataset_statistics.json 파일이 없습니다!")
        print("   먼저 이 파일을 생성해주세요.")
        exit(1)

    # 전체 탐색
    load_and_explore_statistics()

    # Un-normalization 실습
    demonstrate_unnormalization()

    # 정규화 방법 비교
    compare_normalization_methods()

    # 실전 연습
    practical_exercise()

    print("\n" + "=" * 70)
    print("✅ 완료!")
    print("=" * 70)
    print("\n다음 단계:")
    print("  1. 실제 OpenVLA 모델의 dataset_statistics.json 다운로드")
    print("  2. practice_action_tokens.py와 함께 사용")
    print("  3. 실제 모델 추론 시 un-normalization 적용")
