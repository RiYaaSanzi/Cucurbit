# Cucurbit Leaf Disease Detector — Setup Guide

## Project Structure

```
cucurbit_app/
├── app.py                          ← Flask backend
├── requirements.txt                ← Python dependencies
├── EfficientNetV2_SE_ViT_best.pth  ← Your trained model  ← PLACE HERE
├── templates/
│   └── index.html                  ← Web frontend
└── uploads/                        ← Auto-created at runtime
```

---

## Step-by-Step Setup in VS Code

### 1. Open the project folder

```
File → Open Folder → select cucurbit_app/
```

---

### 2. Copy your model file

Place **`EfficientNetV2_SE_ViT_best.pth`** directly inside `cucurbit_app/`
(same folder as `app.py`).

---

### 3. Create a virtual environment

Open the VS Code terminal (`Ctrl + `` ` ```) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of the terminal prompt.

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (recommended for speed):**
> Install the CUDA-enabled PyTorch first, then the rest:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install flask timm opencv-python Pillow numpy
> ```

---

### 5. Run the Flask server

```bash
python app.py
```

Expected terminal output:

```
============================================================
  Cucurbit Leaf Disease Detection System
  Open http://127.0.0.1:5000 in your browser
============================================================

[INFO] Using device: cpu        ← or cuda if GPU available
[INFO] Model loaded successfully from EfficientNetV2_SE_ViT_best.pth
 * Running on http://0.0.0.0:5000
```

---

### 6. Open the app

Visit **http://127.0.0.1:5000** in any browser.

---

## Using the App

1. **Drag & drop** a cucurbit leaf image onto the upload zone, or click to browse.
2. Click **Analyse**.
3. The app returns:
   - Predicted disease / healthy class
   - Confidence score with a visual bar
   - Severity level, description, and treatment recommendation
   - Top-5 alternative predictions

---

## Supported Crops & Conditions

| Crop | Conditions detected |
|------|---------------------|
| Bitter Gourd | Downy Mildew, Mosaic Virus, Healthy |
| Bottle Gourd | Downy Mildew, Mosaic Virus, Healthy |
| Cucumber | Downy Mildew, Mosaic Virus, Healthy |
| Pumpkin | Downy Mildew, Mosaic Virus, Healthy |
| Ridge Gourd | Downy Mildew, Mosaic Virus, Healthy |
| Snake Gourd | Downy Mildew, Mosaic Virus, Healthy |
| Watermelon | Downy Mildew, Mosaic Virus, Healthy |
| Wax Gourd | Downy Mildew, Mosaic Virus, Healthy |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not loaded` in status bar | Check `.pth` file is in `cucurbit_app/` folder |
| `ModuleNotFoundError: timm` | Run `pip install timm` |
| `Address already in use` | Change port: `app.run(port=5001)` in `app.py` |
| Slow on CPU | Use a GPU machine, or reduce image resolution |

---

## API Reference

### `POST /predict`

**Request:** `multipart/form-data` with field `file` (image)

**Response JSON:**
```json
{
  "success": true,
  "predicted": "Cucumber Mosaic Virus",
  "raw_class": "Cucumber_Mosaic_Virus",
  "confidence": 97.43,
  "severity": "High",
  "description": "Viral disease causing mosaic-like patterns...",
  "treatment": "Remove infected plants...",
  "color": "#e74c3c",
  "top5": [
    {"class": "Cucumber Mosaic Virus", "confidence": 97.43},
    ...
  ]
}
```

### `GET /status`

Returns model status, device, and class list.
