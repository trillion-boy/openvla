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
    - Automatic fallback to eager attention if Flash Attention fails
    - Better error messages for common Colab issues
    - Memory optimization hints
    """
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Determine attention implementation
    attn_implementation = "flash_attention_2"

    # Check if Flash Attention is available
    try:
        import flash_attn
        print("[*] Flash Attention 2 detected - will try to use it")
    except ImportError:
        print("[!] Flash Attention 2 not installed - using eager attention (slower but compatible)")
        attn_implementation = "eager"

    # Try loading with Flash Attention first, fall back to eager if it fails
    vla = None
    load_attempts = []

    if attn_implementation == "flash_attention_2":
        print("[*] Attempting to load with Flash Attention 2...")
        try:
            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_checkpoint,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.load_in_8bit,
                load_in_4bit=cfg.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("[✓] Successfully loaded with Flash Attention 2!")
        except Exception as e:
            print(f"[!] Flash Attention 2 failed: {e}")
            print("[*] Falling back to eager attention...")
            load_attempts.append(("flash_attention_2", str(e)))
            attn_implementation = "eager"

    if vla is None:
        # Load with eager attention
        print("[*] Loading with eager attention (compatible mode)...")
        try:
            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_checkpoint,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.load_in_8bit,
                load_in_4bit=cfg.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("[✓] Successfully loaded with eager attention!")
        except Exception as e:
            print(f"[✗] Failed to load model: {e}")
            print("\n" + "="*80)
            print("TROUBLESHOOTING TIPS:")
            print("="*80)
            print("1. If you're using a T4 GPU, try adding --load_in_8bit True")
            print("2. Make sure you have the correct transformers version (4.40.1)")
            print("3. Try restarting your Colab runtime and re-running setup")
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
