"""
Pre-download all AI models for Ultimate Chat.
Run this once to cache all models locally.
"""

import os
import sys

def main():
    print("=" * 60)
    print("🤖 Ultimate Chat - Model Downloader")
    print("=" * 60)
    print()
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Detected: {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("⚠️  No CUDA GPU detected - models will run on CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        return
    
    print()
    
    # Download SDXL
    print("-" * 60)
    print("📥 Downloading Stable Diffusion XL (Image Generation)...")
    print("-" * 60)
    try:
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        
        print("  → Downloading SDXL VAE (for better quality)...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        print("  ✅ VAE downloaded")
        
        print("  → Downloading SDXL Base model (~7GB)...")
        print("     This may take 5-15 minutes depending on your connection...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        print("  ✅ SDXL Base model downloaded")
        
        # Clean up memory
        del pipe, vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Image generation model ready!")
        
    except Exception as e:
        print(f"❌ Failed to download SDXL: {e}")
    
    print()
    
    # Download Stable Video Diffusion
    print("-" * 60)
    print("📥 Downloading Stable Video Diffusion (Video Generation)...")
    print("-" * 60)
    try:
        from diffusers import StableVideoDiffusionPipeline
        
        print("  → Downloading SVD model (~10GB)...")
        print("     This may take 10-20 minutes depending on your connection...")
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        print("  ✅ SVD model downloaded")
        
        del svd_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Video generation model ready!")
        
    except Exception as e:
        print(f"❌ Failed to download SVD: {e}")
    
    print()
    print("=" * 60)
    print("🎉 Model download complete!")
    print()
    print("Models are cached in: ~/.cache/huggingface/hub/")
    print("You can now run Ultimate Chat with instant model loading.")
    print("=" * 60)

if __name__ == "__main__":
    main()

