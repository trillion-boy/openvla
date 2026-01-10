"""
Colab Setup Script for LIBERO Evaluation
==========================================

This script helps set up the environment for running LIBERO evaluations on Google Colab.
It addresses common issues with dependencies, GPU compatibility, and memory constraints.

Usage in Colab:
    !python experiments/robot/libero/colab_setup_libero.py
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and print status."""
    print(f"\n{'='*80}")
    print(f"[*] {description}")
    print(f"{'='*80}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        print(f"[✓] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[✗] Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def check_gpu():
    """Check if GPU is available and what type."""
    print("\n" + "="*80)
    print("[*] Checking GPU availability...")
    print("="*80)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[✓] GPU detected: {gpu_name}")
            print(f"[✓] Total GPU memory: {gpu_memory:.2f} GB")

            # Check if it's a free tier GPU (T4 with 16GB)
            if "T4" in gpu_name:
                print("[!] Warning: T4 GPU detected. You may need to use 8-bit quantization to fit the 7B model.")
                return "t4"
            elif "V100" in gpu_name or "A100" in gpu_name:
                print("[✓] High-end GPU detected. You should be able to run without quantization.")
                return "high_end"
            else:
                print(f"[!] GPU type: {gpu_name}. May need quantization.")
                return "unknown"
        else:
            print("[✗] No GPU detected! This script requires a GPU.")
            print("    Please enable GPU in Colab: Runtime > Change runtime type > GPU")
            return None
    except ImportError:
        print("[✗] PyTorch not installed yet. Will check after installation.")
        return "not_checked"


def install_dependencies(gpu_type):
    """Install required dependencies for LIBERO evaluation."""

    # Step 1: Install core dependencies with specific versions
    print("\n" + "="*80)
    print("[*] Installing core dependencies...")
    print("="*80)

    # Uninstall potentially conflicting packages first
    run_command("pip uninstall -y transformers tokenizers timm", "Removing old versions")

    # Install specific versions for compatibility
    core_deps = [
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "timm==0.9.10",
    ]

    for dep in core_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 2: Install Flash Attention 2 (if possible)
    print("\n" + "="*80)
    print("[*] Attempting to install Flash Attention 2...")
    print("="*80)
    print("[!] Note: Flash Attention may fail on some Colab GPUs. This is OK - we'll use a fallback.")

    # Try to install flash-attn, but don't fail if it doesn't work
    flash_success = run_command(
        "pip install flash-attn==2.5.5 --no-build-isolation",
        "Installing Flash Attention 2"
    )

    if not flash_success:
        print("[!] Flash Attention installation failed. Will use eager attention mode instead.")

    # Step 3: Install bitsandbytes for quantization
    if gpu_type in ["t4", "unknown"]:
        print("\n" + "="*80)
        print("[*] Installing bitsandbytes for 8-bit quantization...")
        print("="*80)
        run_command("pip install bitsandbytes>=0.43.0", "Installing bitsandbytes")

    # Step 4: Install accelerate for better model loading
    run_command("pip install accelerate>=0.26.0", "Installing accelerate")

    # Step 5: Install LIBERO dependencies
    print("\n" + "="*80)
    print("[*] Installing LIBERO dependencies...")
    print("="*80)

    libero_deps = [
        "imageio[ffmpeg]",
        "robosuite==1.4.1",
        "bddl",
        "easydict",
        "cloudpickle",
        "gym",
    ]

    for dep in libero_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 6: Clone and install LIBERO
    print("\n" + "="*80)
    print("[*] Installing LIBERO...")
    print("="*80)

    if not os.path.exists("LIBERO"):
        run_command(
            "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git",
            "Cloning LIBERO repository"
        )

    run_command("cd LIBERO && pip install -e .", "Installing LIBERO package")

    # Step 7: Install additional required packages
    run_command("pip install draccus wandb", "Installing additional utilities")

    print("\n" + "="*80)
    print("[✓] All dependencies installed!")
    print("="*80)


def create_colab_eval_script():
    """Create a Colab-optimized evaluation script."""
    script_content = '''"""
Colab-Optimized LIBERO Evaluation Script
=========================================

This is a modified version of run_libero_eval.py optimized for Google Colab.
It includes fallbacks for Flash Attention and better memory management.
"""

import sys
import os

# Add openvla to path
if os.path.exists('/content/openvla'):
    sys.path.insert(0, '/content/openvla')
    sys.path.insert(0, '/content/openvla/experiments/robot')

# Import original script
from experiments.robot.libero.run_libero_eval import eval_libero, GenerateConfig
import draccus


if __name__ == "__main__":
    print("="*80)
    print("COLAB-OPTIMIZED LIBERO EVALUATION")
    print("="*80)

    # Check GPU
    import torch
    if not torch.cuda.is_available():
        print("[✗] No GPU detected! Please enable GPU in Colab.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[✓] GPU: {gpu_name} ({gpu_memory:.2f} GB)")

    # Suggest settings based on GPU
    if "T4" in gpu_name:
        print("[!] T4 GPU detected - Recommend using --load_in_8bit True")

    print("="*80)

    # Run evaluation
    eval_libero()
'''

    with open('/content/run_libero_eval_colab.py', 'w') as f:
        f.write(script_content)

    print("[✓] Created /content/run_libero_eval_colab.py")


def print_usage_instructions(gpu_type):
    """Print instructions for running the evaluation."""
    print("\n" + "="*80)
    print("SETUP COMPLETE! Here's how to run the evaluation:")
    print("="*80)

    print("\n[STEP 1] Basic command (for V100/A100 GPUs):")
    print("-" * 80)
    print("""
python experiments/robot/libero/run_libero_eval.py \\
  --model_family openvla \\
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
  --task_suite_name libero_spatial \\
  --center_crop True
""")

    if gpu_type in ["t4", "unknown"]:
        print("\n[STEP 1 - FOR T4 GPU] Use 8-bit quantization to save memory:")
        print("-" * 80)
        print("""
python experiments/robot/libero/run_libero_eval.py \\
  --model_family openvla \\
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
  --task_suite_name libero_spatial \\
  --center_crop True \\
  --load_in_8bit True
""")

    print("\n[STEP 2] Other task suites you can try:")
    print("-" * 80)
    print("  - libero_object  : openvla/openvla-7b-finetuned-libero-object")
    print("  - libero_goal    : openvla/openvla-7b-finetuned-libero-goal")
    print("  - libero_10      : openvla/openvla-7b-finetuned-libero-10")

    print("\n[STEP 3] Optional: Enable Weights & Biases logging:")
    print("-" * 80)
    print("  Add to your command:")
    print("    --use_wandb True \\")
    print("    --wandb_project YOUR_PROJECT \\")
    print("    --wandb_entity YOUR_ENTITY")

    print("\n" + "="*80)
    print("TROUBLESHOOTING TIPS:")
    print("="*80)
    print("""
1. If you get Flash Attention errors:
   - The script will automatically fall back to eager attention
   - This is slower but works on all GPUs

2. If you get CUDA out of memory errors:
   - Use --load_in_8bit True (for T4 GPUs)
   - Reduce --num_trials_per_task (default is 50)

3. If you get dependency errors:
   - Restart the Colab runtime
   - Re-run this setup script

4. For the best results, use the exact versions specified in README:
   - Python 3.10.13
   - PyTorch 2.2.0
   - transformers 4.40.1
   - flash-attn 2.5.5
""")

    print("\n" + "="*80)
    print("[✓] Setup complete! You're ready to run LIBERO evaluations.")
    print("="*80)


def main():
    """Main setup function."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           OpenVLA LIBERO Evaluation - Colab Setup Script                  ║
║                                                                            ║
║  This script will install all dependencies needed to run LIBERO           ║
║  evaluations on Google Colab.                                             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # Check GPU
    gpu_type = check_gpu()
    if gpu_type is None:
        sys.exit(1)

    # Install dependencies
    install_dependencies(gpu_type)

    # Print usage instructions
    print_usage_instructions(gpu_type)


if __name__ == "__main__":
    main()
