# Auto-generated inference script
def generate_inference_script():
    return f"""import torch
import json
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os

# ---------------- CONFIG ----------------
IMAGE_SIZE = 224
MODEL_PATH = "best_model.pth"
CLASS_MAPPING_PATH = "class_mapping.json"
# ----------------------------------------

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load class mapping --------
with open(CLASS_MAPPING_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {{int(v): k for k, v in class_to_idx.items()}}
num_classes = len(idx_to_class)

model = torch.load(
    "best_model.pth",
    map_location=device,
    weights_only=False
)
model.to(device)
model.eval()


# -------- Inference transform --------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Model loaded successfully!")
print("Enter image paths to predict.\\n")

# -------- Prediction loop --------
while True:
    choice = input("Predict on an image? (y/n): ").lower().strip()

    if choice != "y":
        print("Exiting inference. Bye!")
        break

    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("Image not found. Try again.")
        continue

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    class_name = idx_to_class[pred_idx.item()]
    confidence = confidence.item() * 100

    print(f"\\nPredicted Class: {{class_name}}")
    print(f"Confidence: {{confidence:.2f}}%\\n")
"""