# 확장된 VLA 학습 로드맵 🚀

멘토님 피드백 반영 + TACO 연구 준비

---

## 🎯 전체 학습 구조

```
Week 1: Action Pipeline ← 이미 시작함!
├─ ✅ Action tokens 추출
├─ ✅ Un-normalization
└─ 🆕 RLDS 데이터 포맷

Week 2: Data & Real-time Constraints
├─ 🆕 RLDS 데이터 로더
├─ 🆕 Control Frequency (5-10Hz)
├─ 🆕 Action Chunking (없음을 이해)
└─ Vision Encoder

Week 3: TACO Integration
├─ Autoregressive Generation
├─ LogitsProcessor
└─ TACO 제약 구현
```

---

## 📅 Week 1: Action Pipeline + RLDS (7일)

### Day 1-2: Action Tokens (완료!)
- [x] `practice_action_tokens.py` 실행
- [x] Un-normalization 수식 이해
- [x] Dataset statistics 찾기

### Day 3-5: RLDS 데이터 포맷 ⭐⭐⭐⭐⭐

#### 목표:
"RLDS가 뭔지, OpenVLA가 어떻게 로드하는지" 이해

#### 학습 내용:

**1. RLDS란?**
- Robot Learning Dataset Standard
- TensorFlow Datasets (tfds) 기반
- 구조: `Dataset → Episodes → Steps → {observations, actions, ...}`

**예시 구조:**
```python
{
  'episode_0': {
    'steps': [
      {
        'observation': {
          'image': [224, 224, 3],
          'state': [7],  # EEF pose + gripper
        },
        'action': [7],
        'language_instruction': 'pick up the cup',
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
      },
      # ... more steps
    ]
  },
  'episode_1': { ... }
}
```

**2. OpenVLA의 RLDS 로더**

**핵심 파일:**
```
prismatic/vla/datasets/rlds/
├── dataset.py              ← make_dataset_from_rlds()
├── utils/
│   └── data_utils.py       ← get_dataset_statistics()
└── oxe/
    ├── configs.py          ← bridge_orig 설정
    ├── transforms.py       ← Dataset별 변환
    └── materialize.py      ← OXE dataset 설정
```

**3. 실습: RLDS 데이터 로드**

파일: `/home/user/openvla/practice_rlds_loading.py`

```python
"""
RLDS 데이터 로딩 실습

목표:
1. RLDS 데이터셋 구조 이해
2. OpenVLA의 데이터 로더 사용법
3. Episode → Steps → Observations/Actions 추출
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS


# ============================================================
# Step 1: RLDS 데이터셋 구조 탐색
# ============================================================

def explore_rlds_structure(dataset_name: str = "bridge_dataset"):
    """
    RLDS 데이터셋의 구조를 출력합니다.

    주의: 실제 데이터가 없을 수 있으므로,
          구조 이해가 목적입니다.
    """
    print("=" * 60)
    print(f"RLDS Dataset: {dataset_name}")
    print("=" * 60)

    # OpenVLA의 설정에서 가져오기
    if "bridge_orig" in OXE_DATASET_CONFIGS:
        config = OXE_DATASET_CONFIGS["bridge_orig"]
        print("\n[Config]")
        print(f"  Image keys: {config.get('image_obs_keys')}")
        print(f"  State keys: {config.get('state_obs_keys')}")
        print(f"  Action encoding: {config.get('action_encoding')}")

    # RLDS 표준 구조
    print("\n[Standard RLDS Structure]")
    print("""
    Dataset
    └── Episodes (trajectories)
        └── Steps (transitions)
            ├── observation
            │   ├── image_0: [H, W, 3]
            │   ├── image_1: [H, W, 3] (optional)
            │   └── state: [state_dim]
            ├── action: [action_dim]
            ├── language_instruction: str
            ├── is_first: bool
            ├── is_last: bool
            └── is_terminal: bool
    """)


# ============================================================
# Step 2: OpenVLA 데이터 로더 사용
# ============================================================

def understand_data_pipeline():
    """
    OpenVLA가 RLDS 데이터를 어떻게 처리하는지 이해
    """
    print("\n" + "=" * 60)
    print("OpenVLA Data Pipeline")
    print("=" * 60)

    print("""
    [Step 1] RLDS 로드
    ├─ make_dataset_from_rlds()
    └─ TensorFlow Dataset 생성

    [Step 2] Dataset-specific Transform
    ├─ bridge_orig_dataset_transform()
    │  ├─ Action 변환 (gripper binarization)
    │  ├─ State 분리 (EEF vs gripper)
    │  └─ Action relabeling
    └─ Output: 표준화된 형식

    [Step 3] Normalization
    ├─ get_dataset_statistics()
    │  └─ Compute q01, q99 for actions
    └─ normalize_action_and_proprio()
       └─ action → [-1, 1]

    [Step 4] Action Tokenization
    ├─ ActionTokenizer(action)
    └─ 연속값 → 256 bins → token IDs

    [Step 5] Prompt 생성
    └─ "What action ... ? ASSISTANT: <tokens>"
    """)


# ============================================================
# Step 3: Bridge Dataset 예제
# ============================================================

def bridge_dataset_example():
    """
    Bridge 데이터셋의 실제 구조
    """
    print("\n" + "=" * 60)
    print("Bridge Dataset 예제")
    print("=" * 60)

    print("""
    [Episode 예시]

    Task: "pick up the blue block"

    Step 0:
      observation:
        image_0: [256, 256, 3]  ← 3인칭 카메라
        image_1: [256, 256, 3]  ← 다른 각도
        state: [7]              ← [x, y, z, roll, pitch, yaw, gripper]
      action: [7]               ← [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]
      language_instruction: "pick up the blue block"
      is_first: True

    Step 1:
      observation: { ... }      ← 로봇이 조금 움직인 후
      action: [7]
      is_first: False

    ...

    Step N:
      observation: { ... }      ← 물체를 잡음
      action: [7]
      is_last: True
      is_terminal: True


    [Action 의미]
    - action[0:3]: EEF의 XYZ 델타 이동 (meters)
    - action[3:6]: Roll-Pitch-Yaw 델타 회전 (radians)
    - action[6]:   Gripper 명령 (0=close, 1=open)

    [중요!]
    - Actions는 **상대값(delta)**: "현재 위치에서 얼마나 움직일지"
    - Gripper는 **절대값**: "열림/닫힘 상태"
    - 이게 absolute_action_mask = [False]*6 + [True]인 이유!
    """)


# ============================================================
# Step 4: RLDS → OpenVLA 변환 과정
# ============================================================

def transformation_pipeline():
    """
    RLDS 원본 → OpenVLA 입력 변환 과정
    """
    print("\n" + "=" * 60)
    print("Transformation Pipeline")
    print("=" * 60)

    print("""
    [1] 원본 RLDS (Bridge)
    {
      'observation': {
        'image': [256, 256, 3],
        'state': [7]  # [x, y, z, r, p, y, gripper]
      },
      'action': [7],  # [Δx, Δy, Δz, Δr, Δp, Δy, gripper_continuous]
      'language_instruction': 'pick up the cup'
    }

    ↓ bridge_orig_dataset_transform()

    [2] 변환 후
    {
      'observation': {
        'image_0': [224, 224, 3],         ← Resize
        'image_1': [224, 224, 3],
        'EEF_state': [6],                 ← state[:6]
        'gripper_state': [1],             ← state[-1:]
      },
      'action': [7],                      ← Gripper binarized
      'task': {
        'language_instruction': 'pick up the cup'
      }
    }

    ↓ normalize_action_and_proprio()

    [3] 정규화
    {
      'action': [-0.3, 0.5, ..., 1.0],   ← [-1, 1] 범위
      ...
    }

    ↓ ActionTokenizer

    [4] 토큰화
    "What action should the robot take to pick up the cup?\nASSISTANT: <tok_1><tok_2>...<tok_7>"
    """)


# ============================================================
# Step 5: 직접 해보기
# ============================================================

def exercise_understanding_rlds():
    """
    연습 문제: RLDS 구조 이해
    """
    print("\n" + "=" * 60)
    print("연습 문제")
    print("=" * 60)

    print("""
    [문제 1] Episode vs Step

    Q: Bridge 데이터셋에서 1개 episode는 몇 개의 steps로 구성되나?
    A: 평균 50-100 steps (README에 "50 episodes per task" 언급)

    Q: 각 step은 몇 Hz로 수집되었나?
    A: 5-10Hz (README의 "control frequency" 참고)

    Q: 따라서 1개 episode는 약 몇 초짜리 데모인가?
    A: 50 steps ÷ 5Hz = 10초 정도


    [문제 2] Action 구조

    다음 RLDS step이 주어졌을 때:

    {
      'observation': {'state': [0.5, 0.3, 0.2, 0, 0, 0, 0.0]},
      'action': [0.01, -0.02, 0.0, 0, 0, 0, 1.0]
    }

    Q1: 로봇의 현재 EEF 위치는?
    A1: (x=0.5, y=0.3, z=0.2)

    Q2: 다음 step에서 로봇은 어디로 이동하나?
    A2: (x=0.51, y=0.28, z=0.2)  # action은 delta!

    Q3: Gripper는 어떻게 되나?
    A3: 열림 (1.0 = open)


    [문제 3] TACO 연결

    Q: TACO로 "X축으로 5cm만 이동" 제약을 걸 때,
       RLDS의 어느 필드를 제어해야 하나?

    A: action[0] (X축 delta)
       - 정규화 공간에서 0.05m에 해당하는 값으로 logits 조정
       - 하지만 다른 차원(Y, Z, rotation)은 자유롭게
    """)


# ============================================================
# Main
# ============================================================

def main():
    """전체 실습 실행"""
    explore_rlds_structure()
    understand_data_pipeline()
    bridge_dataset_example()
    transformation_pipeline()
    exercise_understanding_rlds()

    print("\n" + "=" * 60)
    print("다음 단계")
    print("=" * 60)
    print("""
    1. 실제 RLDS 데이터 다운로드 (선택사항):
       - Bridge V2: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

    2. OpenVLA 코드 읽기:
       - prismatic/vla/datasets/rlds/dataset.py:204-251
       - prismatic/vla/datasets/rlds/oxe/transforms.py:61-86

    3. 다음 주제로:
       - Control Frequency (5-10Hz)
       - Action Chunking
    """)


if __name__ == "__main__":
    main()
