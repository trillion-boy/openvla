# VLA 코딩 학습 로드맵 🚀

초보자를 위한 OpenVLA 코드베이스 탐험 가이드

---

## 🎯 학습 목표

1. **7 action tokens 추출**: 모델 출력 → 실제 로봇 명령 변환 과정 이해
2. **VLA 구조 파악**: Vision-Language-Action 모델의 데이터 흐름 이해
3. **TACO 적용 준비**: Logits 제어 시 normalization 공간 이해

---

## 📚 단계별 학습 경로

### **Phase 1: Action Pipeline 이해** (1-2일)

가장 직관적이고 실용적인 부분부터 시작합니다.

#### Step 1.1: Token 변환 실습 ✅ (가장 쉬움!)
```bash
# 방금 만든 파일 실행
python practice_action_tokens.py
```

**학습 내용:**
- [ ] 7개 action token이 무엇인지
- [ ] Token ID → 연속 값 변환 (binning 개념)
- [ ] 정규화/역정규화 수식

**핵심 파일:**
- `practice_action_tokens.py` ← 방금 만든 실습 파일
- `prismatic/models/action_tokenizer.py:40-80` ← 실제 구현

**디버깅 팁:**
```python
# 중간값 출력해서 확인하기
print(f"Generated IDs shape: {generated_ids.shape}")
print(f"Last 7 tokens: {generated_ids[0, -7:]}")
print(f"Vocab size: {vla.config.vocab_size}")
```

---

#### Step 1.2: Dataset Statistics 찾기
```bash
# 학습 시 생성되는 통계 파일 위치 확인
find ~/.cache/orca -name "dataset_statistics*.json" 2>/dev/null
find . -name "dataset_statistics.json" 2>/dev/null
```

**학습 내용:**
- [ ] q01, q99가 뭔지 (1% / 99% quantile)
- [ ] 왜 mean/std가 아니라 quantile을 쓰는지 (outlier 제거)
- [ ] Bridge dataset의 실제 action 범위

**핵심 파일:**
- `prismatic/vla/datasets/rlds/utils/data_utils.py:185-293`
  - `get_dataset_statistics()` 함수
  - `NormalizationType.BOUNDS_Q99` 정의

**실험해보기:**
```python
import json
import numpy as np

# Statistics 로드
with open("path/to/dataset_statistics.json") as f:
    stats = json.load(f)

bridge_stats = stats["bridge_orig"]["action"]
print(f"q01: {bridge_stats['q01']}")
print(f"q99: {bridge_stats['q99']}")
print(f"Action range: {np.array(bridge_stats['q99']) - np.array(bridge_stats['q01'])}")

# 정규화 변환 테스트
def normalize(action, q01, q99):
    return 2 * (action - q01) / (q99 - q01) - 1

# 예: X축 10cm 이동이 정규화 공간에서 얼마인지?
real_action = 0.10  # 10cm in meters
norm_action = normalize(real_action, bridge_stats['q01'][0], bridge_stats['q99'][0])
print(f"10cm → normalized: {norm_action}")
```

---

#### Step 1.3: Inference Pipeline 따라가기
```bash
# 모델 추론 예제 실행
python experiments/bridge/verify_openvla.py
```

**학습 내용:**
- [ ] `predict_action()` 함수 내부 흐름
- [ ] `unnorm_key="bridge_orig"` 파라미터 역할
- [ ] 전체 pipeline: Image → Tokens → Actions

**핵심 파일:**
- `prismatic/models/openvla.py:61-103` ← `predict_action()` 구현
- `experiments/bridge/verify_openvla.py:84` ← 사용 예제

**코드 리딩 순서:**
```python
# 1. 입력 준비 (verify_openvla.py:70-78)
inputs = processor(prompt, image)

# 2. 토큰 생성 (openvla.py:69-77)
generated_ids = self.generate(**inputs, max_new_tokens=action_dim)

# 3. Action 토큰 추출 (openvla.py:84)
action_token_ids = generated_ids[0, -action_dim:]

# 4. 정규화 action 복원 (openvla.py:87-89)
normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(...)

# 5. Un-normalization (openvla.py:94-103)
action_stats = self.get_action_stats(unnorm_key)
actions = 0.5 * (normalized_actions + 1) * (high - low) + low
```

---

### **Phase 2: Training Pipeline 이해** (2-3일)

데이터가 어떻게 모델로 들어가는지 역추적합니다.

#### Step 2.1: Dataset Transform 이해
**학습 내용:**
- [ ] Bridge dataset이 어떻게 변환되는지
- [ ] Gripper action binarization
- [ ] EEF state vs gripper state 분리

**핵심 파일:**
- `prismatic/vla/datasets/rlds/oxe/transforms.py:61-86`
  - `bridge_orig_dataset_transform()` 함수

**실험:**
```python
# transforms.py의 변환 로직 따라해보기
import tensorflow as tf

# 원본 데이터 (예시)
raw_action = tf.constant([[0.1, -0.2, 0.05, 0.0, 0.0, 0.1, 0.6]])  # 7D

# Gripper binarization
gripper_continuous = raw_action[:, -1]  # 0.6
gripper_binary = tf.where(gripper_continuous > 0.5, 1.0, -1.0)  # → 1.0 (open)

print(f"원본 gripper: {gripper_continuous.numpy()}")
print(f"Binary gripper: {gripper_binary.numpy()}")
```

---

#### Step 2.2: Normalization 과정 추적
**학습 내용:**
- [ ] BOUNDS_Q99 정규화 방식
- [ ] Action mask (gripper는 정규화 안함!)
- [ ] 왜 gripper는 특별 취급하는지

**핵심 파일:**
- `prismatic/vla/datasets/rlds/utils/data_utils.py:61-103`
  - `normalize_action_and_proprio()` 함수
- `prismatic/vla/datasets/rlds/oxe/materialize.py:35-42`
  - `action_normalization_mask` 설정

**중요 개념:**
```python
# Gripper는 이미 0-1 범위로 표준화되어 있음
action_normalization_mask = [True, True, True, True, True, True, False]
                             # ↑ EEF 6개 차원만 정규화      ↑ Gripper는 그대로

# 정규화 (EEF만)
normalized_action[:6] = 2 * (action[:6] - q01[:6]) / (q99[:6] - q01[:6]) - 1
normalized_action[6] = action[6]  # Gripper는 변경 없음
```

---

#### Step 2.3: Action Tokenization
**학습 내용:**
- [ ] 연속 값을 discrete token으로 변환하는 이유
- [ ] 256 bins의 의미
- [ ] Vocabulary의 마지막 256개를 왜 쓰는지

**핵심 파일:**
- `prismatic/models/action_tokenizer.py`
- `prismatic/vla/datasets/datasets.py:40-49` ← 학습 중 사용

**실험:**
```python
from prismatic.models.action_tokenizer import ActionTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b")
action_tokenizer = ActionTokenizer(tokenizer)

# 연속 action → 토큰
action = np.array([0.5, -0.3, 0.8, 0.0, -1.0, 1.0, 0.9])
tokens = action_tokenizer(action)
print(f"Action: {action}")
print(f"Tokens: {tokens}")

# 토큰 → 연속 action (복원)
# ... (실제 token IDs 필요)
```

---

### **Phase 3: TACO 적용 준비** (1-2일)

이제 TACO를 어떻게 통합할지 생각합니다.

#### Step 3.1: Logits 추출 위치 파악
**학습 내용:**
- [ ] Action tokens의 logits가 어디서 나오는지
- [ ] 생성 과정에서 logits 접근 방법
- [ ] Autoregressive generation (7개 토큰을 순차 생성)

**핵심 포인트:**
```python
# OpenVLA는 autoregressive하게 7개 토큰을 생성
# 각 step에서:
#   logits = model(context)[vocab_size]  # 전체 vocabulary에 대한 확률
#   action_logits = logits[-256:]        # 마지막 256개만 action용

# TACO 적용 시:
#   1. 어느 차원의 토큰을 생성 중인지 확인 (1/7, 2/7, ...)
#   2. 해당 차원의 목표값을 정규화 공간으로 변환
#   3. Logits 조정
```

**코드 예시 (pseudocode):**
```python
# TACO 제약: "X축으로 10cm 이동"
target_real = 0.10  # meters
target_norm = normalize(target_real, q01[0], q99[0])  # → 예: 0.35

# 정규화 값 → bin index
target_bin = int((target_norm + 1) / 2 * 256)  # 0.35 → bin 173

# 생성 중 logits 조정
for step in range(7):
    logits = model(...)

    if step == 0:  # X축 차원
        # target_bin 근처 logits 강화 (TACO loss)
        logits = apply_taco_constraint(logits, target_bin)

    next_token = sample(logits)
```

---

#### Step 3.2: Multi-step Generation Hook
**학습 내용:**
- [ ] `generate()` 함수의 내부 구조
- [ ] `GenerationMixin` 커스터마이징
- [ ] Logits processor 사용법

**참고 파일:**
- HuggingFace Transformers의 `generation/utils.py`
- `LogitsProcessor` 클래스 상속

**예제:**
```python
from transformers import LogitsProcessor

class TACOLogitsProcessor(LogitsProcessor):
    def __init__(self, constraints, action_tokenizer, stats):
        self.constraints = constraints
        self.action_tokenizer = action_tokenizer
        self.stats = stats
        self.current_action_dim = 0

    def __call__(self, input_ids, scores):
        # 현재 어느 action 차원을 생성 중인지 추적
        if self.current_action_dim < 7:
            # 해당 차원의 제약 적용
            constraint = self.constraints[self.current_action_dim]
            scores = self.apply_constraint(scores, constraint)
            self.current_action_dim += 1
        return scores

# 사용
vla.generate(
    **inputs,
    logits_processor=[TACOLogitsProcessor(...)],
)
```

---

## 🔍 핵심 파일 요약

### **Inference (추론)**
| 파일 | 역할 | 중요도 |
|------|------|--------|
| `prismatic/models/openvla.py` | `predict_action()` - 전체 추론 pipeline | ⭐⭐⭐⭐⭐ |
| `prismatic/models/action_tokenizer.py` | Token ↔ Action 변환 | ⭐⭐⭐⭐⭐ |
| `experiments/bridge/verify_openvla.py` | 사용 예제 | ⭐⭐⭐⭐ |

### **Training (학습)**
| 파일 | 역할 | 중요도 |
|------|------|--------|
| `prismatic/vla/datasets/datasets.py` | Dataset loading + action tokenization | ⭐⭐⭐⭐ |
| `prismatic/vla/datasets/rlds/utils/data_utils.py` | Normalization + statistics | ⭐⭐⭐⭐⭐ |
| `prismatic/vla/datasets/rlds/oxe/transforms.py` | Dataset-specific transforms | ⭐⭐⭐ |
| `prismatic/vla/datasets/rlds/oxe/materialize.py` | Dataset configs | ⭐⭐⭐ |

### **Configuration**
| 파일 | 역할 | 중요도 |
|------|------|--------|
| `prismatic/vla/datasets/rlds/oxe/configs.py` | `bridge_orig` 등 설정 | ⭐⭐⭐⭐ |
| `prismatic/vla/datasets/rlds/oxe/mixtures.py` | Multi-dataset mixing | ⭐⭐ |

---

## 🎓 학습 체크리스트

### Week 1: Action Pipeline
- [ ] `practice_action_tokens.py` 실행 성공
- [ ] 연습 문제 1, 2 풀이
- [ ] Dataset statistics JSON 파일 찾기
- [ ] `verify_openvla.py` 코드 리딩
- [ ] 직접 이미지로 추론 실행

### Week 2: Training Pipeline
- [ ] `bridge_orig_dataset_transform()` 이해
- [ ] Normalization 수식 손으로 계산
- [ ] Action tokenization 실험
- [ ] Gripper 특수 처리 이유 설명 가능

### Week 3: TACO Integration
- [ ] Logits processor 구현
- [ ] 정규화 공간에서 제약 걸기
- [ ] Multi-step generation hook
- [ ] 간단한 TACO 제약 테스트

---

## 💡 자주 하는 실수

### 1. **정규화 공간 혼동**
❌ 잘못된 예:
```python
# "10cm 이동" 제약을 실제 값으로 걸기
target = 0.10  # meters
logits = apply_constraint(logits, target)  # 🚫 틀림!
```

✅ 올바른 예:
```python
# 먼저 정규화 공간으로 변환
target_real = 0.10
target_norm = 2 * (target_real - q01) / (q99 - q01) - 1
target_bin = int((target_norm + 1) / 2 * 256)
logits = apply_constraint(logits, target_bin)  # ✅ 맞음!
```

### 2. **Gripper 정규화**
❌ 잘못된 예:
```python
# Gripper도 [-1, 1]로 정규화한다고 착각
normalized_gripper = (gripper - q01[6]) / (q99[6] - q01[6])  # 🚫 틀림!
```

✅ 올바른 예:
```python
# Gripper는 이미 [0, 1] 또는 {-1, 1} (binary)
# 정규화 하지 않음!
normalized_gripper = gripper  # ✅ 그대로 사용
```

### 3. **Token ID 범위**
❌ 잘못된 예:
```python
# Action tokens이 vocabulary 앞부분에 있다고 착각
action_token_ids = generated_ids[0, :7]  # 🚫 틀림!
```

✅ 올바른 예:
```python
# 마지막 256개가 action tokens
# 생성된 sequence의 마지막 7개를 추출
action_token_ids = generated_ids[0, -7:]  # ✅ 맞음!
```

---

## 📖 추가 학습 자료

### Paper
- **OpenVLA**: "Open-Source Vision-Language-Action Models"
- **RT-1**: "Robotics Transformer" (action tokenization 기법)
- **Octo**: "Open X-Embodiment" (normalization 방법론)

### Code Reference
- HuggingFace Transformers: `generation/utils.py`
- TACO 원본 구현 (있다면 링크)

### Debug 명령어
```bash
# 모델 구조 확인
python -c "from prismatic.models import load_vla; vla = load_vla('openvla/openvla-7b'); print(vla)"

# Tokenizer vocab size 확인
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('openvla/openvla-7b'); print(f'Vocab: {len(t)}')"

# Dataset stats 확인
find . -name "dataset_statistics.json" -exec cat {} \; | python -m json.tool
```

---

## 🚀 다음 단계

이 로드맵을 완료하면:
1. VLA의 전체 데이터 흐름 이해 완료
2. TACO 통합을 위한 코드 수정 위치 파악
3. 정규화 공간에서의 제약 설계 가능

**멘토와 논의할 주제:**
- TACO loss를 어느 단계에서 적용할지
- Multi-step generation에서 autoregressive TACO
- 실험 설계 (어떤 task로 검증할지)

---

Good luck! 🎉
