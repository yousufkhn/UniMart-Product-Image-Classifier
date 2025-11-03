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

## Project commands (based on repo scripts)

All runnable scripts live under the `scripts/` folder. They are implemented to work when run from the repository root. Example usages (PowerShell):

- Split the raw dataset into train/val/test (creates `data_split/`):
```powershell
python scripts/split_dataset.py
```

- Preprocess images in-place (resize / remove corrupt files):
```powershell
python scripts/preprocess_images.py
```

- Download images from Unsplash (requires `UNSPLASH_ACCESS_KEY` in `.env`):
```powershell
python scripts/dataset_downloader.py
```

- Train the model (uses `src.get_data_loaders` and `src/model_builder`):
```powershell
python scripts/train_model.py --epochs 10 --batch_size 32 --lr 0.001 --num_workers 0
```

Notes:
- `scripts/train_model.py` uses `get_data_loaders(data_dir="data_split")` (defined in `src/data_loader.py`) and constructs a ResNet-18 via `src/model_builder.build_model()`.
- The script will create a `checkpoints/` directory and save the best model to `checkpoints/best_model.pth`.
- `num_workers` defaults to `0` (safe for Windows). Increase when training on Linux with more CPU workers.

---

## Predict (single image)

Use `scripts/predict.py` to run a single-image prediction. It expects two files produced by training:
- `checkpoints/best_model.pth` — saved model weights
- `checkpoints/best_model_classes.json` — mapping index→class

Run (PowerShell):
```powershell
python scripts/predict.py
```

The script loads the model using `src.model_builder.build_model` and performs standard ImageNet-style preprocessing.

---

## Files and key functions (quick reference)

- `src/data_loader.py` — get_data_loaders(data_dir="data_split", batch_size=32, num_workers=0)
- `src/model_builder.py` — build_model(num_classes)
- `src/model_utils.py` — save_checkpoint(...), accuracy(...)
- `scripts/train_model.py` — training entrypoint (CLI args: --epochs, --batch_size, --lr, --num_workers, --data_dir)
- `scripts/predict.py` — single-image prediction
- `scripts/split_dataset.py` — prepare `data_split/` from `dataset/`
- `scripts/dataset_downloader.py` — downloads images from Unsplash (requires API key)

---

## Troubleshooting notes

- Module imports: scripts insert the repository root into `sys.path` so you can run scripts directly with `python scripts/<name>.py`. If you prefer not to rely on that, run modules with `-m` or set `PYTHONPATH`.
- If you see `FileNotFoundError` for checkpoints or classes when running `predict.py`, confirm you have trained the model and the `checkpoints/` directory contains `best_model.pth` and `best_model_classes.json`.
- On Windows prefer `--num_workers 0` to avoid DataLoader worker start issues.

---

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