# scripts/predict.py
import torch
from torchvision import transforms
from PIL import Image
import json
import os
from api.src.model_builder import build_model

# --- config ---
MODEL_PATH = "checkpoints/best_model.pth"
CLASS_PATH = "checkpoints/best_model_classes.json"
IMAGE_PATH = "dataset/random/luggage.jpg"   # change this to any image you want
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load model + classes ---
with open(CLASS_PATH, "r") as f:
    classes = json.load(f)

model = build_model(num_classes=len(classes))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(DEVICE)

# --- image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- predict ---
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
    pred_class = classes[preds.item()]
    return pred_class

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    prediction = predict_image(IMAGE_PATH)
    print(f"Predicted category: {prediction}")
