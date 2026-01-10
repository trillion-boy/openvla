"""
Colab-Optimized Utils for Evaluating OpenVLA Policy
====================================================

This is a modified version of openvla_utils.py that includes fallbacks for
Flash Attention and better error handling for Colab environments.
"""

import json
import os
import time
import warnings

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """
    Loads and returns a VLA model from checkpoint with Colab-optimized settings.

    This version includes:
    - SDPA (Scaled Dot Product Attention) as primary fallback - RECOMMENDED for T4 GPUs!
    - Automatic attention implementation selection with smart fallbacks
    - Better error messages for common Colab issues
    - Memory optimization hints

    Attention implementation priority:
    1. Flash Attention 2 (fastest, but may not work on T4)
    2. SDPA (RECOMMENDED - fast, compatible, no bugs) ⭐
    3. Eager (slowest, has tensor size bugs - avoid if possible)
    """
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("="*80)

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Check PyTorch version for SDPA support
    pytorch_version = torch.__version__.split("+")[0]
    has_sdpa = tuple(map(int, pytorch_version.split("."))) >= (2, 0, 0)

    if has_sdpa:
        print("[✓] PyTorch 2.0+ detected - SDPA is available!")
    else:
        print(f"[!] PyTorch {pytorch_version} detected - SDPA requires 2.0+")

    # Check if Flash Attention is available
    has_flash_attn = False
    try:
        import flash_attn
        has_flash_attn = True
        print("[✓] Flash Attention 2 is installed")
    except ImportError:
        print("[!] Flash Attention 2 not installed")

    # Determine attention implementation priority
    # Try: Flash Attention 2 -> SDPA -> Eager
    attention_priority = []
    if has_flash_attn:
        attention_priority.append("flash_attention_2")
    if has_sdpa:
        attention_priority.append("sdpa")  # RECOMMENDED!
    attention_priority.append("eager")  # Last resort

    print(f"[*] Attention implementation priority: {' -> '.join(attention_priority)}")
    print("="*80)

    # Additional config for 8-bit quantization
    quantization_config = None
    if cfg.load_in_8bit or cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            if cfg.load_in_8bit:
                print("[*] Configuring 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            elif cfg.load_in_4bit:
                print("[*] Configuring 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        except ImportError:
            print("[!] WARNING: bitsandbytes not installed. Quantization may fail.")

    # Try loading with each attention implementation in priority order
    vla = None
    load_attempts = []

    for attn_impl in attention_priority:
        print(f"\n[*] Attempting to load with {attn_impl} attention...")
        try:
            load_kwargs = {
                "attn_implementation": attn_impl,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

            # Add quantization config if specified
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"  # Required for quantization to avoid .to() calls
            elif cfg.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"  # Required for 8-bit quantization
            elif cfg.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"  # Required for 4-bit quantization

            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_checkpoint,
                **load_kwargs
            )

            print(f"[✓] Successfully loaded with {attn_impl} attention!")
            if attn_impl == "sdpa":
                print("    🎯 SDPA is the recommended mode for Colab T4 GPUs!")
            elif attn_impl == "eager":
                print("    ⚠️  WARNING: Eager mode may have tensor size bugs!")
                print("    ⚠️  If you get 'size of tensor a (291) must match (290)' errors,")
                print("    ⚠️  try upgrading transformers or using a different checkpoint.")
            break

        except Exception as e:
            error_msg = str(e)
            load_attempts.append((attn_impl, error_msg))
            print(f"[✗] {attn_impl} failed: {error_msg[:100]}...")

            # Don't continue if this is the last option
            if attn_impl == attention_priority[-1]:
                print("\n" + "="*80)
                print("ALL ATTENTION IMPLEMENTATIONS FAILED!")
                print("="*80)
                print("Attempted implementations:")
                for impl, err in load_attempts:
                    print(f"  - {impl}: {err[:80]}...")
                print("\nTROUBLESHOOTING TIPS:")
                print("="*80)
                print("1. If you're using a T4 GPU, try adding --load_in_8bit True")
                print("2. Make sure you have the correct transformers version (4.40.1)")
                print("3. Check your PyTorch version (should be 2.2.0)")
                print("4. Try restarting your Colab runtime and re-running setup")
                print("5. Try: pip install bitsandbytes>=0.43.0 --upgrade")
                print("="*80)
                raise

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        print(f"[*] Moving model to {DEVICE}")
        vla = vla.to(DEVICE)
    else:
        quantization_type = "8-bit" if cfg.load_in_8bit else "4-bit"
        print(f"[*] Model loaded with {quantization_type} quantization (already on correct device)")

    # Print memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[*] GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
        print("[✓] Loaded dataset statistics for action un-normalization")
    else:
        # Try downloading from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            print("[*] Attempting to download dataset_statistics.json from HuggingFace Hub...")
            local_path = hf_hub_download(
                repo_id=cfg.pretrained_checkpoint,
                filename="dataset_statistics.json",
                cache_dir=None,
            )
            with open(local_path, "r") as f:
                norm_stats = json.load(f)
            vla.norm_stats = norm_stats
            print("[✓] Downloaded and loaded dataset statistics")
        except Exception as e:
            print(
                f"[!] WARNING: Could not load dataset_statistics.json: {e}\n"
                "    This is OK if you're using the base (non-fine-tuned) VLA checkpoint.\n"
                "    Otherwise, you may encounter errors when calling `predict_action()`."
            )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    print("[*] Loading model processor...")
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    print("[✓] Processor loaded successfully")
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
