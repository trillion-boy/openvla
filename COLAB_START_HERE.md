# 🚀 Colab에서 OpenVLA LIBERO 평가 시작하기

**여기서 시작하세요!** 이 가이드는 Google Colab에서 처음부터 끝까지 실행하는 방법을 설명합니다.

---

## ✅ 빠른 체크리스트

Colab에서 실행하기 전에 다음을 확인하세요:

- [ ] Google Colab 노트북 열기
- [ ] GPU 활성화: **Runtime → Change runtime type → GPU (T4 선택)**
- [ ] 아래의 코드 셀을 순서대로 실행

---

## 📝 Step-by-Step 가이드

### Step 1: GPU 확인

먼저 새 Colab 노트북을 만들고 GPU가 할당되었는지 확인:

```python
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}")
    if "T4" in gpu_name:
        print("⚠️ T4 detected - Use --load_in_8bit True")
else:
    print("❌ NO GPU! Please enable GPU in Runtime settings")
```

### Step 2: 저장소 클론 (중요!)

**여러분의 fork**를 클론하고 Colab 최적화 브랜치를 체크아웃:

```bash
# 1. 저장소 클론
!git clone https://github.com/trillion-boy/openvla.git
%cd openvla

# 2. Colab 최적화 브랜치로 체크아웃
!git fetch origin
!git checkout claude/libero-spatial-eval-setup-Xhupi

# 3. 현재 브랜치 확인 (claude/libero-spatial-eval-setup-Xhupi 이어야 함)
!git branch --show-current
```

### Step 3: 환경 설정

자동 설정 스크립트 실행 (모든 dependency 설치):

```bash
!python experiments/robot/libero/colab_setup_libero.py
```

이 스크립트는 다음을 자동으로 수행합니다:
- GPU 타입 감지
- PyTorch, transformers, bitsandbytes 설치
- LIBERO 설치
- 필요한 모든 패키지 설치

### Step 4: Colab 최적화 활성화 ⭐ 중요!

텐서 크기 버그 (291 vs 290)를 피하기 위해 SDPA 버전 사용:

```bash
!cp experiments/robot/openvla_utils_colab.py experiments/robot/openvla_utils.py
```

### Step 5: 런타임 재시작 ⚠️

**필수!** 새 패키지를 설치했으므로 런타임을 재시작해야 합니다:

1. **Runtime → Restart runtime** 클릭
2. Step 6부터 다시 실행

### Step 6: LIBERO Evaluation 실행

런타임 재시작 후:

```bash
# 작업 디렉토리로 이동
%cd /content/openvla

# T4 GPU의 경우 (8비트 양자화 사용)
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10
```

**참고**: `--num_trials_per_task 10`은 빠른 테스트용입니다. 논문 결과 재현을 위해서는 `50`으로 설정하세요.

### Step 7: 결과 확인

```bash
# 최신 로그 확인
!tail -30 $(ls -t experiments/logs/*.txt | head -1)

# 생성된 비디오 확인
!ls -lh rollouts/$(date +%Y_%m_%d)/*.mp4 | head -10
```

---

## 🎯 다른 Task Suites 실행

### LIBERO-Object

```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10
```

### LIBERO-Goal

```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10
```

### LIBERO-10 (Long Horizon)

```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10
```

---

## ⚠️ 흔한 오류 및 해결책

### 오류 1: 텐서 크기 불일치 (291 vs 290)

```
Caught exception: The size of tensor a (291) must match the size of tensor b (290)
```

**원인**: Eager attention 모드의 버그

**해결책**: Step 4를 빠뜨렸는지 확인! 다음 명령어 다시 실행:
```bash
!cp experiments/robot/openvla_utils_colab.py experiments/robot/openvla_utils.py
```

### 오류 2: CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결책**: T4 GPU에서는 반드시 `--load_in_8bit True` 사용

### 오류 3: 8비트 양자화 오류

```
RuntimeError: CUDA error: no kernel image is available
```

**해결책**: bitsandbytes 재설치
```bash
!pip uninstall -y bitsandbytes transformers
!pip install bitsandbytes>=0.43.0
!pip install transformers==4.40.1
# 런타임 재시작 후 Step 6부터 다시 실행
```

### 오류 4: 브랜치를 찾을 수 없음

```
error: pathspec 'claude/libero-spatial-eval-setup-Xhupi' did not match any file(s)
```

**해결책**: fetch를 먼저 실행
```bash
!git fetch origin
!git checkout claude/libero-spatial-eval-setup-Xhupi
```

---

## 🎓 이 브랜치의 개선 사항

`claude/libero-spatial-eval-setup-Xhupi` 브랜치에는 다음이 포함되어 있습니다:

1. **SDPA Fallback** ⭐
   - Flash Attention → SDPA → Eager 순서로 자동 시도
   - 텐서 크기 버그 (291 vs 290) 해결
   - T4 GPU에서 완벽 호환

2. **개선된 8비트 양자화**
   - BitsAndBytesConfig 자동 설정
   - 더 나은 오류 처리
   - 호환성 문제 해결

3. **자동 설정 스크립트**
   - `colab_setup_libero.py` - 모든 dependency 자동 설치
   - GPU 타입 자동 감지
   - 적절한 설정 제안

4. **완전한 문서화**
   - `COLAB_LIBERO_GUIDE.md` - 상세 문제 해결 가이드 (한국어/영어)
   - `COLAB_LIBERO_QUICKSTART.ipynb` - 실행 가능한 Jupyter 노트북
   - `COLAB_START_HERE.md` (이 파일) - 빠른 시작 가이드

---

## 📊 성능 비교

| Attention Mode | T4 호환성 | 속도 | 버그 | 권장 |
|----------------|----------|------|------|------|
| Flash Attention 2 | ❌ 낮음 | 100% | ✅ 없음 | A100용 |
| **SDPA** ⭐ | ✅ 완벽 | 70-80% | ✅ 없음 | **T4용 (권장!)** |
| Eager | ✅ 호환 | 40-50% | ❌ 있음 | 피하기 |

---

## 💾 결과 저장 (선택사항)

Google Drive에 결과를 저장하려면:

```python
from google.colab import drive
drive.mount('/content/drive')

# 로그 및 비디오 복사
!mkdir -p /content/drive/MyDrive/openvla_results
!cp -r experiments/logs/* /content/drive/MyDrive/openvla_results/
!cp -r rollouts/* /content/drive/MyDrive/openvla_results/

print("✅ Results saved to Google Drive!")
```

---

## 📚 추가 리소스

- **상세 가이드**: [COLAB_LIBERO_GUIDE.md](COLAB_LIBERO_GUIDE.md)
- **Jupyter 노트북**: [COLAB_LIBERO_QUICKSTART.ipynb](COLAB_LIBERO_QUICKSTART.ipynb)
- **GitHub 저장소**: https://github.com/trillion-boy/openvla/tree/claude/libero-spatial-eval-setup-Xhupi
- **OpenVLA 논문**: https://arxiv.org/abs/2406.09246
- **LIBERO 프로젝트**: https://libero-project.github.io/

---

## ❓ 도움이 필요하신가요?

1. 먼저 [COLAB_LIBERO_GUIDE.md](COLAB_LIBERO_GUIDE.md)의 Troubleshooting 섹션을 확인하세요
2. 문제가 계속되면 GitHub Issues에 보고해주세요
3. 구체적인 오류 메시지와 GPU 타입을 포함해주세요

---

**Happy Evaluating! 🚀**

*마지막 업데이트: 2026-01-10*
