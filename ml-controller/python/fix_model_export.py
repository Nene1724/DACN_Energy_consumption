import timm
import torch
import os

model_name = "mobilenetv3_small_050"
model_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_store")
os.makedirs(model_store, exist_ok=True)

print(f"Downloading {model_name}...")
model = timm.create_model(model_name, pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
output_path = os.path.join(model_store, f"{model_name}.onnx")

print("Exporting to ONNX with embedded weights...")
# Force all weights to be embedded in single file
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Check if external data was created
data_file = output_path + ".data"
if os.path.exists(data_file):
    print(f"WARNING: External data file created: {data_file}")
    print("This won't work for deployment. Trying alternative method...")
    
    # Remove both files
    os.remove(output_path)
    os.remove(data_file)
    
    # Try with save_as_external_data=False (PyTorch older method)
    import torch.onnx
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"✅ Success! Size: {size_mb:.2f} MB")
print(f"Location: {output_path}")

if os.path.exists(output_path + ".data"):
    print("❌ Still has .data file!")
else:
    print("✓ Single file ONNX (weights embedded)")
