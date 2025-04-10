import torch
from model import get_model

# Load the trained model
model = get_model(pretrained=True)  # Match your training setup
state_dict = torch.load('../outputs/run_20250406_234429/checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
model.eval()

# Define dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Shape: (batch_size, channels, height, width)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "../outputs/run_20250406_234429/checkpoints/model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11  # Stable opset; adjust if needed
)

print("Model exported to model.onnx")