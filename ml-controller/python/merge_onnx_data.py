import onnx
import os

model_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_store")
onnx_path = os.path.join(model_store, "mobilenetv3_small_050.onnx")
data_path = onnx_path + ".data"

print("Loading ONNX model with external data...")
model = onnx.load(onnx_path, load_external_data=True)

print("Converting external data to embedded...")
output_path = os.path.join(model_store, "mobilenetv3_small_050_embedded.onnx")

# Save with embedded data
onnx.save(model, output_path, save_as_external_data=False)

# Remove old files
os.remove(onnx_path)
os.remove(data_path)

# Rename to original name
os.rename(output_path, onnx_path)

size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"✅ Success! Embedded model: {size_mb:.2f} MB")
print(f"Location: {onnx_path}")

# Verify no .data file
if os.path.exists(data_path):
    print("❌ .data file still exists!")
else:
    print("✓ Single file ONNX (all weights embedded)")
