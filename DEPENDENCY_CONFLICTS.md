# 🔧 Dependency Conflicts 해결 가이드

Google Colab에서 OpenVLA LIBERO evaluation을 실행할 때 발생하는 의존성 충돌 문제와 해결 방법을 설명합니다.

---

## ⚠️ 흔한 의존성 오류

Colab에서 `colab_setup_libero.py`를 실행하면 다음과 같은 경고가 나타날 수 있습니다:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
This behaviour is the source of the following dependency conflicts.

sentence-transformers 5.2.0 requires transformers<6.0.0,>=4.41.0,
  but you have transformers 4.40.1 which is incompatible.

torchvision 0.24.0+cu126 requires torch==2.9.0,
  but you have torch 2.2.0 which is incompatible.

torchaudio 2.9.0+cu126 requires torch==2.9.0,
  but you have torch 2.2.0 which is incompatible.
```

---

## 🔍 문제 분석

### 1. PyTorch 버전 충돌 ⚠️ 가장 심각!

| 패키지 | OpenVLA 요구 | Colab 기본 설치 | 충돌 |
|--------|-------------|----------------|------|
| **torch** | 2.2.0 | 2.9.0 | ❌ |
| **torchvision** | 0.17.0 (torch 2.2.0 호환) | 0.24.0 (torch 2.9.0 요구) | ❌ |
| **torchaudio** | 2.2.0 (torch 2.2.0 호환) | 2.9.0 (torch 2.9.0 요구) | ❌ |

**문제**:
- OpenVLA는 PyTorch 2.2.0을 요구 (논문 재현 및 flash-attn 2.5.5 호환성)
- Colab은 최신 PyTorch 2.9.0을 pre-install
- torch를 2.2.0으로 다운그레이드하면, torchvision/torchaudio가 2.9.0 버전으로 남아서 버전 불일치

**왜 PyTorch 2.2.0이 필요한가?**
- flash-attn 2.5.5는 PyTorch 2.2.0과 가장 잘 호환됨
- 논문 재현을 위해 정확한 버전 필요
- PyTorch 2.9.0에서는 일부 API가 변경되어 호환성 문제 발생 가능

### 2. transformers 버전 충돌

| 패키지 | OpenVLA 요구 | Colab 기본 설치 | 충돌 |
|--------|-------------|----------------|------|
| **transformers** | 4.40.1 (정확한 버전) | >=4.41.0 (sentence-transformers 요구) | ❌ |
| **tokenizers** | 0.19.1 | (설치 안됨) | ⚠️ |

**문제**:
- OpenVLA는 transformers 4.40.1을 요구 (모델 호환성)
- Colab의 `sentence-transformers` 5.2.0이 transformers >=4.41.0을 요구
- transformers를 4.40.1로 다운그레이드하면 sentence-transformers와 충돌

**왜 transformers 4.40.1이 필요한가?**
- OpenVLA 모델이 transformers 4.40.1에서 테스트됨
- 최신 버전에서는 API 변경으로 인한 호환성 문제 발생 가능
- `PrismaticProcessor` 등의 custom code가 4.40.1에 맞춰져 있음

### 3. 기타 패키지 충돌

```
peft 0.18.0 requires transformers, which is not installed.
torchtune 0.6.1 requires tokenizers, which is not installed.
```

**문제**:
- Colab pre-installed packages가 transformers/tokenizers를 require
- 우리가 uninstall 하면서 이들 패키지가 broken dependencies를 가지게 됨

---

## ✅ 해결 방법 (이미 적용됨)

`colab_setup_libero.py`가 이제 다음과 같이 의존성을 해결합니다:

### Step 0: Conflicting Packages 제거

```python
conflicting_packages = [
    "transformers",
    "tokenizers",
    "timm",
    "sentence-transformers",  # ← 중요! transformers 4.40.1과 충돌
    "torchvision",            # ← 중요! torch 2.2.0과 호환되는 버전 설치 필요
    "torchaudio",             # ← 중요! torch 2.2.0과 호환되는 버전 설치 필요
]
```

### Step 1: PyTorch Ecosystem 일괄 설치

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

**중요**: PyTorch, torchvision, torchaudio는 반드시 호환되는 버전으로 **함께** 설치해야 합니다!

| torch | torchvision | torchaudio |
|-------|------------|-----------|
| 2.2.0 | 0.17.0 | 2.2.0 |

### Step 2: Transformers Ecosystem 설치

```bash
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10
```

### Step 3-8: 나머지 패키지 설치

- Flash Attention 2 (optional)
- bitsandbytes (T4 GPU용)
- accelerate
- LIBERO dependencies
- LIBERO 자체
- 추가 유틸리티

---

## 🤔 "의존성 충돌 경고가 여전히 뜨는데?"

경고가 나타나도 **대부분 괜찮습니다!** 다음을 확인하세요:

### ✅ 무시해도 되는 경고:

```
sentence-transformers 5.2.0 requires transformers<6.0.0,>=4.41.0,
  but you have transformers 4.40.1
```
→ **OK**: sentence-transformers를 사용하지 않으므로 문제없음

```
peft 0.18.0 requires transformers, which is not installed.
```
→ **OK**: transformers 4.40.1이 설치되어 있음 (pip가 잘못 인식)

```
torchtune 0.6.1 requires tokenizers, which is not installed.
```
→ **OK**: tokenizers 0.19.1이 설치되어 있음 (pip가 잘못 인식)

### ⚠️ 주의해야 할 경고:

```
torchvision X.X.X requires torch==Y.Y.Y, but you have torch 2.2.0
```
→ **문제**: PyTorch ecosystem 버전 불일치. 설정 스크립트를 다시 실행하세요.

```
ModuleNotFoundError: No module named 'transformers'
```
→ **문제**: transformers가 제대로 설치되지 않음. 런타임 재시작 후 다시 시도.

---

## 🔧 수동 수정 방법

만약 자동 설정 스크립트가 실패하면, 다음 순서로 수동 설치:

```bash
# 1. Conflicting packages 제거
!pip uninstall -y transformers tokenizers timm sentence-transformers torchvision torchaudio

# 2. PyTorch ecosystem 설치 (함께 설치 중요!)
!pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# 3. Transformers ecosystem 설치
!pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

# 4. 런타임 재시작
# Runtime → Restart runtime

# 5. 설치 확인
import torch
import transformers
print(f"PyTorch: {torch.__version__}")           # 2.2.0이어야 함
print(f"Transformers: {transformers.__version__}") # 4.40.1이어야 함
```

---

## 📊 버전 호환성 표

### PyTorch Ecosystem

| torch | torchvision | torchaudio | flash-attn | 호환성 |
|-------|------------|-----------|-----------|--------|
| 2.2.0 | 0.17.0 | 2.2.0 | 2.5.5 | ✅ OpenVLA 권장 |
| 2.0.0 | 0.15.0 | 2.0.0 | 2.3.0 | ⚠️ 오래됨 |
| 2.9.0 | 0.24.0 | 2.9.0 | N/A | ❌ OpenVLA 미지원 |

### Transformers Ecosystem

| transformers | tokenizers | timm | 호환성 |
|-------------|-----------|------|--------|
| 4.40.1 | 0.19.1 | 0.9.10 | ✅ OpenVLA 권장 |
| 4.41.0+ | 0.19.1 | 0.9.10 | ⚠️ API 변경 가능 |
| 4.30.0 | 0.13.0 | 0.9.0 | ❌ 너무 오래됨 |

---

## 💡 왜 이렇게 복잡한가?

1. **Python 패키지 생태계의 한계**:
   - pip는 모든 패키지의 의존성을 동시에 해결할 수 없음
   - 충돌하는 요구사항이 있으면 경고만 표시

2. **Colab의 Pre-installed Packages**:
   - Colab은 범용적인 최신 패키지를 pre-install
   - OpenVLA는 재현성을 위해 특정 버전 요구
   - 이 둘이 충돌함

3. **CUDA 호환성**:
   - PyTorch, torchvision, torchaudio는 CUDA 버전과도 맞아야 함
   - flash-attn도 특정 PyTorch 버전과 CUDA 버전이 필요
   - 모든 조합을 맞추기 어려움

---

## 🎯 결론

**경고가 나타나도 괜찮습니다!** 다음을 확인하세요:

✅ **성공 체크리스트**:
- [ ] PyTorch 2.2.0이 설치되었나?
- [ ] transformers 4.40.1이 설치되었나?
- [ ] SDPA가 활성화되었나? (openvla_utils_colab.py 복사)
- [ ] 모델이 로딩되나?
- [ ] Evaluation이 실행되나?

위 항목이 모두 ✅라면, 의존성 경고는 무시해도 됩니다!

❌ **문제 체크리스트**:
- [ ] `ModuleNotFoundError: No module named 'transformers'`
- [ ] `torch.cuda.OutOfMemoryError`
- [ ] `size of tensor a (291) must match (290)`

위 항목 중 하나라도 발생하면:
1. `COLAB_LIBERO_GUIDE.md`의 Troubleshooting 참조
2. 런타임 재시작 후 설정 스크립트 다시 실행
3. 여전히 안 되면 수동 설치 방법 시도

---

## 📚 참고 자료

- [PyTorch Version Compatibility](https://pytorch.org/get-started/previous-versions/)
- [Transformers Version Compatibility](https://github.com/huggingface/transformers/releases)
- [Flash Attention Installation](https://github.com/Dao-AILab/flash-attention)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246) - Appendix에 정확한 버전 명시

---

**마지막 업데이트**: 2026-01-10
**상태**: ✅ 프로덕션 준비 완료
