# Google Colab에서 LIBERO Evaluation 실행 가이드
# Guide for Running LIBERO Evaluation on Google Colab

[한국어](#한국어-가이드) | [English](#english-guide)

---

## 한국어 가이드

### 🎯 개요

이 가이드는 Google Colab에서 OpenVLA의 LIBERO Simulation Benchmark Evaluations를 실행할 때 발생하는 일반적인 문제들을 해결하는 방법을 설명합니다.

### ⚠️ 주요 문제점들

Colab에서 LIBERO evaluation을 실행할 때 다음과 같은 문제들이 발생할 수 있습니다:

1. **Flash Attention 2 호환성 문제**
   - Colab의 GPU (특히 T4)에서 Flash Attention 2가 지원되지 않을 수 있음
   - CUDA 버전 불일치로 인한 설치 실패

2. **8비트 양자화 오류**
   - `bitsandbytes` 라이브러리와 transformers 버전 충돌
   - CUDA 커널 로딩 실패

3. **Dependency 호환성 문제**
   - PyTorch, transformers, tokenizers 버전 불일치
   - Colab의 기본 설치 패키지와 충돌

4. **메모리 부족**
   - T4 GPU (16GB)에서 7B 모델 로딩 시 메모리 부족
   - bfloat16으로도 약 14GB 필요

### 🚀 빠른 시작 (Colab에서 실행)

#### 1단계: 저장소 클론 및 설정

```bash
# Colab 노트북에서 실행
!git clone https://github.com/openvla/openvla.git
%cd openvla

# Colab 전용 설정 스크립트 실행
!python experiments/robot/libero/colab_setup_libero.py
```

이 스크립트는 자동으로:
- GPU 타입 감지 (T4, V100, A100 등)
- 적절한 dependency 버전 설치
- Flash Attention 설치 시도 (실패해도 OK)
- LIBERO 및 필수 패키지 설치
- 문제 해결 팁 제공

#### 2단계: Evaluation 실행

**A. V100/A100 GPU의 경우 (양자화 없이):**
```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

**B. T4 GPU의 경우 (8비트 양자화 사용):**
```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True
```

### 🔧 문제 해결

#### Flash Attention 오류가 발생하는 경우

```
ValueError: FlashAttention only support fp16 and bf16 data type
```

**해결책**: 제공된 Colab 최적화 스크립트를 사용하세요. 자동으로 eager attention으로 전환합니다.

```bash
# experiments/robot/openvla_utils.py 대신 Colab 버전 사용
!cp experiments/robot/openvla_utils_colab.py experiments/robot/openvla_utils.py
```

#### 8비트 양자화 오류

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**해결책 1**: bitsandbytes 최신 버전 설치
```bash
!pip install bitsandbytes>=0.43.0 --upgrade
```

**해결책 2**: 양자화 없이 실행 (V100/A100에서만)
```bash
# --load_in_8bit 옵션 제거
```

#### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결책 1**: 8비트 양자화 사용
```bash
--load_in_8bit True
```

**해결책 2**: trials 수 줄이기
```bash
--num_trials_per_task 10  # 기본값 50에서 줄임
```

**해결책 3**: 런타임 재시작 및 메모리 정리
```python
import torch
torch.cuda.empty_cache()
```

#### Dependency 버전 충돌

```
ImportError: cannot import name 'xxx' from 'transformers'
```

**해결책**: 정확한 버전 재설치
```bash
!pip uninstall -y transformers tokenizers timm
!pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10
```

### 📊 다른 Task Suites 실행

```bash
# LIBERO-Object
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --load_in_8bit True

# LIBERO-Goal
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --load_in_8bit True

# LIBERO-10 (Long Horizon)
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --load_in_8bit True
```

### 💡 Colab Pro 팁

1. **GPU 선택**: Runtime > Change runtime type > GPU (T4, V100, or A100)
2. **메모리 사용 모니터링**:
   ```python
   !nvidia-smi
   ```
3. **세션 유지**: Colab이 자동으로 연결을 끊지 않도록 주의
4. **결과 저장**: Google Drive에 마운트하여 로그 저장
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### 🎓 논문 재현을 위한 권장 사항

논문의 결과를 정확히 재현하려면:

- **Python**: 3.10.13
- **PyTorch**: 2.2.0
- **transformers**: 4.40.1
- **flash-attn**: 2.5.5
- **GPU**: NVIDIA A100

⚠️ **주의**: Colab의 무료 T4 GPU에서는 정확한 재현이 어려울 수 있습니다. 결과가 약간 다를 수 있습니다.

### 📝 완전한 Colab 노트북 예제

```python
# Cell 1: 설치
!git clone https://github.com/openvla/openvla.git
%cd openvla
!python experiments/robot/libero/colab_setup_libero.py

# Cell 2: GPU 확인
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Cell 3: Evaluation 실행
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10

# Cell 4: 결과 확인
!cat experiments/logs/*.txt | tail -20
```

---

## English Guide

### 🎯 Overview

This guide explains how to run OpenVLA's LIBERO Simulation Benchmark Evaluations on Google Colab and solve common issues.

### ⚠️ Common Issues

When running LIBERO evaluation on Colab, you may encounter:

1. **Flash Attention 2 Compatibility**
   - Flash Attention 2 may not be supported on Colab GPUs (especially T4)
   - Installation failures due to CUDA version mismatch

2. **8-bit Quantization Errors**
   - Conflicts between `bitsandbytes` library and transformers versions
   - CUDA kernel loading failures

3. **Dependency Compatibility**
   - Version mismatches in PyTorch, transformers, tokenizers
   - Conflicts with Colab's pre-installed packages

4. **Out of Memory**
   - Insufficient memory on T4 GPU (16GB) for 7B model
   - Requires ~14GB even with bfloat16

### 🚀 Quick Start (Run in Colab)

#### Step 1: Clone Repository and Setup

```bash
# Run in Colab notebook
!git clone https://github.com/openvla/openvla.git
%cd openvla

# Run Colab-specific setup script
!python experiments/robot/libero/colab_setup_libero.py
```

This script automatically:
- Detects GPU type (T4, V100, A100, etc.)
- Installs appropriate dependency versions
- Attempts to install Flash Attention (OK if it fails)
- Installs LIBERO and required packages
- Provides troubleshooting tips

#### Step 2: Run Evaluation

**A. For V100/A100 GPUs (without quantization):**
```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

**B. For T4 GPUs (with 8-bit quantization):**
```bash
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True
```

### 🔧 Troubleshooting

#### Flash Attention Errors

```
ValueError: FlashAttention only support fp16 and bf16 data type
```

**Solution**: Use the provided Colab-optimized script. It automatically falls back to eager attention.

```bash
# Replace openvla_utils.py with Colab version
!cp experiments/robot/openvla_utils_colab.py experiments/robot/openvla_utils.py
```

#### 8-bit Quantization Errors

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution 1**: Install latest bitsandbytes
```bash
!pip install bitsandbytes>=0.43.0 --upgrade
```

**Solution 2**: Run without quantization (V100/A100 only)
```bash
# Remove --load_in_8bit flag
```

#### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution 1**: Use 8-bit quantization
```bash
--load_in_8bit True
```

**Solution 2**: Reduce number of trials
```bash
--num_trials_per_task 10  # Reduced from default 50
```

**Solution 3**: Restart runtime and clear memory
```python
import torch
torch.cuda.empty_cache()
```

#### Dependency Version Conflicts

```
ImportError: cannot import name 'xxx' from 'transformers'
```

**Solution**: Reinstall exact versions
```bash
!pip uninstall -y transformers tokenizers timm
!pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10
```

### 📊 Running Other Task Suites

```bash
# LIBERO-Object
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --load_in_8bit True

# LIBERO-Goal
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --load_in_8bit True

# LIBERO-10 (Long Horizon)
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --load_in_8bit True
```

### 💡 Colab Pro Tips

1. **GPU Selection**: Runtime > Change runtime type > GPU (T4, V100, or A100)
2. **Monitor Memory Usage**:
   ```python
   !nvidia-smi
   ```
3. **Keep Session Alive**: Be aware that Colab may disconnect automatically
4. **Save Results**: Mount Google Drive to save logs
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### 🎓 Recommendations for Paper Reproduction

For exact reproduction of paper results:

- **Python**: 3.10.13
- **PyTorch**: 2.2.0
- **transformers**: 4.40.1
- **flash-attn**: 2.5.5
- **GPU**: NVIDIA A100

⚠️ **Note**: Exact reproduction may be difficult on Colab's free T4 GPU. Results may vary slightly.

### 📝 Complete Colab Notebook Example

```python
# Cell 1: Installation
!git clone https://github.com/openvla/openvla.git
%cd openvla
!python experiments/robot/libero/colab_setup_libero.py

# Cell 2: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Cell 3: Run Evaluation
!python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True \
  --num_trials_per_task 10

# Cell 4: View Results
!cat experiments/logs/*.txt | tail -20
```

---

## 🤝 도움이 더 필요하신가요? / Need More Help?

- GitHub Issues: https://github.com/openvla/openvla/issues
- 이 가이드의 문제: 새 이슈를 생성해주세요 / For issues with this guide: Create a new issue

---

## 📄 라이선스 / License

이 가이드는 OpenVLA 프로젝트의 일부로 MIT License 하에 배포됩니다.

This guide is part of the OpenVLA project and distributed under the MIT License.
