#!/usr/bin/env python3

"""
everything u are 4 - Enhanced Edition
Tier 1 Optimizations: Size Tiers, Dynamic Warmup, Adaptive Gradient Accumulation, Model LR Multipliers, Mixed Precision
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


# ============================================================================
# TIER 1 OPTIMIZATIONS - High Impact Performance Enhancements
# ============================================================================

def get_size_tier_enhanced(image_count: int) -> str:
    """Enhanced size tier selection with 7 tiers for better granularity"""
    if 1 <= image_count <= 5:
        return "xxs"  # Extra Extra Small - Very cautious training
    elif 6 <= image_count <= 10:
        return "xs"   # Extra Small
    elif 11 <= image_count <= 20:
        return "s"    # Small
    elif 21 <= image_count <= 30:
        return "m"    # Medium
    elif 31 <= image_count <= 50:
        return "l"    # Large
    elif 51 <= image_count <= 100:
        return "xl"   # Extra Large
    else:  # 100+
        return "xxl"  # Extra Extra Large - Maximum efficiency
    

def calculate_dynamic_warmup_steps(max_train_epochs: int, dataset_size: int, batch_size: int) -> int:
    """Calculate warmup steps as percentage of total training steps"""
    total_steps = (max_train_epochs * dataset_size) // max(batch_size, 1)
    
    # Small datasets need more warmup (10%), large datasets less (5%)
    warmup_ratio = 0.10 if dataset_size <= 20 else 0.05
    
    warmup_steps = max(int(total_steps * warmup_ratio), 10)  # Minimum 10 steps
    print(f"Calculated dynamic warmup: {warmup_steps} steps ({warmup_ratio*100}% of {total_steps} total steps)", flush=True)
    return warmup_steps


def calculate_adaptive_gradient_accumulation(dataset_size: int, batch_size: int, gpu_count: int = 1) -> int:
    """Calculate optimal gradient accumulation based on dataset size and resources"""
    # Target effective batch sizes for optimal gradient quality
    if dataset_size <= 10:
        target_effective_batch = 8
    elif dataset_size <= 30:
        target_effective_batch = 12
    else:
        target_effective_batch = 16
    
    # Calculate accumulation steps to reach target
    current_effective = batch_size * gpu_count
    accumulation = max(1, target_effective_batch // current_effective)
    
    print(f"Adaptive gradient accumulation: {accumulation} steps (target effective batch: {target_effective_batch})", flush=True)
    return accumulation


def get_model_lr_multiplier(model_name: str) -> float:
    """Model-specific learning rate multipliers based on training characteristics"""
    model_lr_multipliers = {
        # Baseline models (1.0)
        "stabilityai/stable-diffusion-xl-base-1.0": 1.0,
        
        # More sensitive models - need lower LR (0.85-0.95)
        "cagliostrolab/animagine-xl-4.0": 0.85,
        "John6666/nova-anime-xl-pony-v5-sdxl": 0.95,
        "KBlueLeaf/Kohaku-XL-Zeta": 0.90,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 0.92,
        
        # Robust models - can handle higher LR (1.10-1.15)
        "SG161222/RealVisXL_V4.0": 1.15,
        "dataautogpt3/ProteusV0.5": 1.10,
        "dataautogpt3/ProteusSigma": 1.12,
        
        # Moderate adjustments (0.95-1.05)
        "Lykon/dreamshaper-xl-1-0": 1.02,
        "dataautogpt3/CALAMITY": 0.98,
        "dataautogpt3/TempestV0.1": 1.05,
        "Corcelio/mobius": 0.97,
    }
    
    multiplier = model_lr_multipliers.get(model_name, 1.0)
    if multiplier != 1.0:
        print(f"Model LR multiplier for '{model_name}': {multiplier}x", flush=True)
    return multiplier


def get_mixed_precision_config(model_type: str, dataset_size: int) -> str:
    """Select optimal mixed precision based on model type and dataset size"""
    # BF16: More stable, better for small datasets and large models
    # FP16: Faster, good for most cases with sufficient data
    
    if model_type in ["flux"]:
        # Flux models are large, prefer bf16 for stability
        precision = "bf16"
    elif dataset_size < 5:
        # Very small datasets need maximum stability
        precision = "bf16"
    else:
        # Standard case: fp16 is faster
        precision = "fp16"
    
    print(f"Selected mixed precision: {precision} (model_type={model_type}, dataset_size={dataset_size})", flush=True)
    return precision


# ============================================================================
# END TIER 1 OPTIMIZATIONS
# ============================================================================


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count



def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path, 0  # Return dataset_size=0 for ai-toolkit (not used)
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        size_config_loaded = False
        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                if model_type == "flux":
                    print(f"Applying model-specific config for Flux model", flush=True)
                    for key, value in lrs_settings.items():
                        config[key] = value
                else:
                    # TIER 1 ENHANCEMENT: Use enhanced 7-tier size selection
                    size_key = get_size_tier_enhanced(dataset_size)
                    
                    if size_key and size_key in lrs_settings:
                        print(f"Applying model-specific config for size '{size_key}' ({dataset_size} images)", flush=True)
                        
                        # Apply base config from size tier
                        for key, value in lrs_settings[size_key].items():
                            config[key] = value
                        size_config_loaded = True
                        
                        # TIER 1 ENHANCEMENT: Apply model-specific LR multiplier
                        lr_multiplier = get_model_lr_multiplier(model_name)
                        if "unet_lr" in config:
                            original_unet_lr = config["unet_lr"]
                            config["unet_lr"] = original_unet_lr * lr_multiplier
                            if lr_multiplier != 1.0:
                                print(f"Adjusted unet_lr: {original_unet_lr} → {config['unet_lr']} (×{lr_multiplier})", flush=True)
                        
                        if "text_encoder_lr" in config:
                            original_te_lr = config["text_encoder_lr"]
                            config["text_encoder_lr"] = original_te_lr * lr_multiplier
                            if lr_multiplier != 1.0:
                                print(f"Adjusted text_encoder_lr: {original_te_lr} → {config['text_encoder_lr']} (×{lr_multiplier})", flush=True)
                        
                        # TIER 1 ENHANCEMENT: Calculate dynamic warmup steps
                        # Only set warmup for schedulers that support it
                        if "max_train_epochs" in config and "train_batch_size" in config:
                            lr_scheduler = config.get("lr_scheduler", "constant")
                            # constant scheduler doesn't support warmup
                            if lr_scheduler != "constant":
                                dynamic_warmup = calculate_dynamic_warmup_steps(
                                    config["max_train_epochs"],
                                    dataset_size,
                                    config["train_batch_size"]
                                )
                                config["lr_warmup_steps"] = dynamic_warmup
                            else:
                                # Ensure no warmup steps for constant scheduler
                                config["lr_warmup_steps"] = 0
                        
                        # TIER 1 ENHANCEMENT: Calculate adaptive gradient accumulation
                        if "train_batch_size" in config:
                            adaptive_grad_accum = calculate_adaptive_gradient_accumulation(
                                dataset_size,
                                config["train_batch_size"],
                                gpu_count=1  # Can be parameterized from environment
                            )
                            config["gradient_accumulation_steps"] = adaptive_grad_accum
                    else:
                        print(f"Warning: No size configuration '{size_key}' found for model '{model_name}'.", flush=True)
                        print(f"Available tiers in config: {list(lrs_settings.keys())}", flush=True)
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": []
            },
            235: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style[model_name]]
            else:
                network_config = config_mapping[network_config_person[model_name]]

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]


        # Old size config search removed as requested
        if dataset_size > 0 and not size_config_loaded:
             print(f"Warning: No size-specific configuration (xs/s/m/l/xl) found for model '{model_name}' with {dataset_size} images. Using model defaults.", flush=True)
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path, dataset_size  # Return dataset_size for mixed precision selection


def run_training(model_type, config_path, dataset_size=50):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    # TIER 1 ENHANCEMENT: Get optimal mixed precision for this training
    mixed_precision = get_mixed_precision_config(model_type, dataset_size)
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", mixed_precision,  # Dynamic precision
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", mixed_precision,  # Dynamic precision
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path, dataset_size = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path, dataset_size)


if __name__ == "__main__":
    asyncio.run(main())