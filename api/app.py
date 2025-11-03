from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import json
import io
import os
from src.model_builder import build_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="UniMart Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
MODEL_PATH = "checkpoints/best_model.pth"
CLASS_PATH = "checkpoints/best_model_classes.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model + classes once ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found.")
if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError("Class file not found.")

with open(CLASS_PATH, "r") as f:
    classes = json.load(f)

model = build_model(num_classes=len(classes))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(DEVICE)

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)
    return classes[pred_idx.item()], float(confidence.item())

# --- Routes ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pred_class, confidence = predict_image(contents)
        return JSONResponse({"prediction": pred_class, "confidence": round(confidence, 3)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
