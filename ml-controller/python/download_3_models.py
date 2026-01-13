import timm
import torch
import onnx
import os

model_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_store")
os.makedirs(model_store, exist_ok=True)

models_to_download = [
    "tf_mobilenetv3_small_minimal_100",
    "mobilenetv3_small_075"
]

for model_name in models_to_download:
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Download model
        print(f"📥 Loading from timm...")
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        # Get input size
        try:
            data_config = timm.data.resolve_data_config(model.pretrained_cfg)
            input_size = data_config.get('input_size', (3, 224, 224))
        except:
            input_size = (3, 224, 224)
        
        print(f"📊 Input size: {input_size}")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Export to ONNX
        temp_output = os.path.join(model_store, f"{model_name}_temp.onnx")
        output_path = os.path.join(model_store, f"{model_name}.onnx")
        
        print(f"🔄 Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            temp_output,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Check for external data
        temp_data_file = temp_output + ".data"
        if os.path.exists(temp_data_file):
            print(f"🔗 Merging external data...")
            onnx_model = onnx.load(temp_output, load_external_data=True)
            onnx.save(onnx_model, output_path, save_as_external_data=False)
            os.remove(temp_output)
            os.remove(temp_data_file)
        else:
            os.rename(temp_output, output_path)
        
        # Verify
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Success!")
        print(f"   📦 {model_name}.onnx")
        print(f"   💾 {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("Summary:")
print(f"{'='*60}")
files = os.listdir(model_store)
for f in sorted(files):
    if f.endswith('.onnx'):
        size_mb = os.path.getsize(os.path.join(model_store, f)) / (1024 * 1024)
        print(f"✓ {f:<50} {size_mb:>8.2f} MB")
