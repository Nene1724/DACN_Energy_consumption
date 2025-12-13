"""
On-demand model downloader service for ML Controller
Downloads specific models only when requested by users via web interface
"""

import timm
import torch
import os
import sys
from pathlib import Path


def get_model_store_dir():
    """Get model_store directory path (one level up from python/)"""
    return Path(__file__).parent.parent / "model_store"


def download_model(model_name, output_dir=None):
    """
    Download a specific model from timm and save as .pth file
    
    Args:
        model_name: Name of model to download (e.g., 'efficientnet_b0')
        output_dir: Optional output directory (defaults to ../model_store)
    
    Returns:
        tuple: (success: bool, message: str, file_path: str or None)
    """
    if output_dir is None:
        output_dir = get_model_store_dir()
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.pth"
    
    # Check if already exists
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        return (True, f"Model already exists ({size_mb:.1f} MB)", str(output_path))
    
    try:
        # Download from timm
        model = timm.create_model(model_name, pretrained=True)
        
        # Save state dict (more compact than full model)
        torch.save(model.state_dict(), output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        return (True, f"Downloaded successfully ({size_mb:.1f} MB)", str(output_path))
        
    except Exception as e:
        # Clean up partial file if exists
        if output_path.exists():
            output_path.unlink()
        return (False, f"Download failed: {str(e)}", None)


def check_model_exists(model_name, output_dir=None):
    """Check if model file already exists"""
    if output_dir is None:
        output_dir = get_model_store_dir()
    else:
        output_dir = Path(output_dir)
    
    model_path = output_dir / f"{model_name}.pth"
    return model_path.exists()


def get_model_info(model_name, output_dir=None):
    """Get information about a model file if it exists"""
    if output_dir is None:
        output_dir = get_model_store_dir()
    else:
        output_dir = Path(output_dir)
    
    model_path = output_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        return None
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    return {
        'name': model_name,
        'path': str(model_path),
        'size_mb': round(size_mb, 2),
        'exists': True
    }


def main():
    """CLI interface for downloading models"""
    if len(sys.argv) < 2:
        print("Usage: python download_models.py <model_name>")
        print("\nExamples:")
        print("  python download_models.py efficientnet_b0")
        print("  python download_models.py mobilenetv3_large_100")
        print("  python download_models.py ghostnet_100")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print(f"Downloading model: {model_name}")
    print(f"Output directory: {get_model_store_dir()}")
    print("-" * 60)
    
    success, message, file_path = download_model(model_name)
    
    print(message)
    if success and file_path:
        print(f"File saved to: {file_path}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
                else:
                    skip_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    print(f"✅ Downloaded: {success_count}")
    print(f"⏭️  Skipped: {skip_count} (already exist)")
    print(f"❌ Failed: {fail_count}")
    print("=" * 70)
    
    # List all .pth files
    pth_files = list(Path(__file__).parent.glob("*.pth"))
    print(f"\n📦 Total .pth files: {len(pth_files)}")
    for f in sorted(pth_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print("\n✨ Done! Models ready for deployment.")

if __name__ == "__main__":
    main()
