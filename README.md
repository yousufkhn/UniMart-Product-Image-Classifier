# 🛍️ UniMart AI — Product Image Classifier

A deep-learning powered image classification model for automatic product categorization in the UniMart marketplace app.

When a user uploads a product photo on UniMart, the backend automatically predicts its category using this model.

Example:
- Upload an image of sneakers → Model predicts “Footwear”

This saves time, reduces manual tagging, and makes the product upload experience seamless.

---

## ✨ Features
- Transfer-learning image classifier (PyTorch).
- Auto-download images from Unsplash/Kaggle (script included).
- Lightweight API for serving predictions (Flask / FastAPI).
- Ready to integrate with UniMart Node backend and React Native app.

---

## 🧠 Tech Stack

| Component      | Technology        |
| -------------- | ----------------- |
| Deep Learning  | PyTorch           |
| API            | Flask / FastAPI   |
| Dataset        | Unsplash / Kaggle |
| Integration    | UniMart Node + React Native |

---

## 📂 Repository structure

```
unimart-ai/
 ├── dataset_downloader.py   # Script to auto-download images
 ├── dataset/                # Images (ignored in git)
 ├── model/                  # Trained model weights
 ├── app/                    # Flask / FastAPI model API
 ├── notebooks/              # Training experiments
 ├── requirements.txt
 └── README.md
```

---

## ⚙️ Setup

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/unimart-ai.git
cd unimart-ai
```

2. Create & activate a virtual environment

- macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

- Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
# or, if using cmd:
# venv\Scripts\activate.bat
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your Unsplash API key to a `.env` file (create `.env` at repo root):
```
UNSPLASH_ACCESS_KEY=your_key_here
```

5. Run the dataset downloader
```bash
python dataset_downloader.py
```

---

## 🧑‍💻 Training (coming soon)

Planned flow:
- Train a transfer learning model (e.g., ResNet-18 / ResNet-50) on the collected dataset.
- Save model weights to `model/model.pth`.
- Export label-to-index mapping (JSON) for inference.

Suggested (future) commands:
```bash
python train.py --config configs/resnet18.yaml
```

---

## 🚀 Serving predictions

Planned API:
- Flask or FastAPI app under `app/` that loads `model/model.pth` and exposes prediction endpoints.
- Example endpoints:
	- `POST /predict` — accept an image and return predicted category + confidence.
	- `GET /health` — basic health check.

Example (when API is implemented):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or for Flask
python app/server.py
```

---

## 🧩 Future integration ideas
- Real-time category suggestions during product upload.
- Smart retraining pipeline using user feedback (active learning).
- Attribute detection (color, brand, material).
- Mobile-optimized model for on-device inference.

---

## ✅ Author

Yousuf Khan  
Student Developer | Founder of UniMart  
Making AI practical, one startup project at a time.

---

## ⚠️ License

MIT License — free to use and modify.