"""
Cucurbit Leaf Disease Detection System
Flask Backend - app.py
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import timm


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")


# Sorted alphabetically — must match sorted(os.listdir(data_dir)) from training
CLASS_NAMES = [
    "Gourd_Alternaria_Blight",
    "Gourd_Anthracnose",
    "Gourd_Downy_Mildew",
    "Gourd_Healthy",
    "Gourd_Mosaic_Virus",
    "Pumpkin_Healthy",
    "Pumpkin_Mosaic_Virus",
    "Pumpkin_Powdery_Mildew",
]
NUM_CLASSES = len(CLASS_NAMES)


DISEASE_INFO = {
    "Alternaria_Blight": {
        "severity": "Moderate–High",
        "description": "Fungal disease (Alternaria spp.) causing dark brown circular spots with yellow halos. Spreads rapidly in warm, humid conditions.",
        "treatment": "Apply mancozeb or iprodione fungicide. Remove infected leaves. Avoid overhead watering. Improve field drainage.",
        "color": "#d97706"
    },
    "Anthracnose": {
        "severity": "High",
        "description": "Fungal disease (Colletotrichum spp.) causing water-soaked lesions turning dark brown/black on leaves, stems, and fruits.",
        "treatment": "Apply copper-based fungicide or chlorothalonil. Remove infected debris. Use disease-free seeds. Ensure good air circulation.",
        "color": "#b45309"
    },
    "Downy_Mildew": {
        "severity": "Moderate–High",
        "description": "Oomycete disease causing yellow angular spots on upper leaf surface with gray/purple sporulation underneath.",
        "treatment": "Apply metalaxyl or mancozeb fungicide. Improve air circulation. Avoid overhead irrigation. Use resistant varieties.",
        "color": "#ea580c"
    },
    "Mosaic_Virus": {
        "severity": "High",
        "description": "Viral disease causing mosaic yellow-green patterns, leaf distortion, blistering, and stunted plant growth.",
        "treatment": "No chemical cure. Remove infected plants immediately. Control aphid vectors. Use virus-free certified seeds.",
        "color": "#dc2626"
    },
    "Powdery_Mildew": {
        "severity": "Moderate",
        "description": "Fungal disease (Podosphaera xanthii) causing white powdery spots on leaf surfaces. Weakens plants and reduces yield.",
        "treatment": "Apply sulfur-based fungicide or potassium bicarbonate. Spray neem oil as organic option. Remove heavily infected leaves.",
        "color": "#9333ea"
    },
    "Healthy": {
        "severity": "None",
        "description": "Leaf appears healthy with no visible disease, pest damage, or nutrient deficiency.",
        "treatment": "Maintain regular watering, balanced fertilisation, and routine preventive pest management.",
        "color": "#16a34a"
    },
}


def get_disease_info(class_name):
    for key in DISEASE_INFO:
        if key in class_name:
            return DISEASE_INFO[key]
    return DISEASE_INFO["Healthy"]


def format_class_name(raw):
    return raw.replace('_', ' ').title()


def get_crop_name(class_name):
    if class_name.startswith("Gourd"):
        return "Gourd"
    elif class_name.startswith("Pumpkin"):
        return "Pumpkin"
    return "Unknown"


# ─── Model Architecture ───────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class EfficientNetV2_SE_ViT_Fusion(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=pretrained, features_only=True)
        feat_channels = self.backbone.feature_info[-1]['num_chs']

        self.se_block    = SEBlock(feat_channels, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_fc    = nn.Sequential(nn.Linear(feat_channels, 384), nn.ReLU(inplace=True))

        # ✅ FIX: renamed from token_proj → token_projection to match checkpoint
        self.token_projection = nn.Conv2d(feat_channels, 768, kernel_size=1)

        self.vit_encoders = nn.ModuleList([
            TransformerEncoderBlock(768, 8, 2048, 0.1) for _ in range(6)])
        self.vit_fc   = nn.Sequential(nn.Linear(768, 384), nn.ReLU(inplace=True))
        self.fusion   = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, num_classes))

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.se_block(feat)

        local_vec = self.global_pool(feat).flatten(1)
        local_vec = self.local_fc(local_vec)

        # ✅ FIX: use token_projection here as well
        tokens = self.token_projection(feat)
        B, C, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)

        for enc in self.vit_encoders:
            tokens = enc(tokens)

        vit_vec = self.vit_fc(tokens.mean(1))
        return self.fusion(torch.cat([local_vec, vit_vec], 1))


# ─── Load Model ───────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'EfficientNetV2_SE_ViT_best.pth')
model = None


def load_model():
    global model
    try:
        m = EfficientNetV2_SE_ViT_Fusion(num_classes=NUM_CLASSES, pretrained=False)
        state = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(state, strict=True)
        m.to(device)
        m.eval()
        model = m
        print(f"[INFO] Model loaded successfully — {NUM_CLASSES} classes — device: {device}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")


load_model()


# ─── Preprocessing ────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = (224, 224)


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def gamma_correction(img, gamma=1.2):
    lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def unsharp_mask(img, sigma=1.0, strength=1.5):
    f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f + strength * (f - blurred), 0, 255).astype(np.uint8)


def preprocess_image(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img = apply_clahe(img)
    img = gamma_correction(img, 1.2)
    img = unsharp_mask(img, 1.0, 1.5)
    t   = img.astype(np.float32) / 255.0
    t   = (t - IMAGENET_MEAN) / IMAGENET_STD
    return torch.FloatTensor(t).permute(2, 0, 1).unsqueeze(0).to(device)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Place EfficientNetV2_SE_ViT_best.pth next to app.py'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}:
        return jsonify({'error': f'.{ext} not supported.'}), 400

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))

        buf = io.BytesIO()
        pil_image.convert('RGB').save(buf, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        tensor = preprocess_image(pil_image)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()

        top_idx   = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf  = float(probs[top_idx]) * 100

        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [
            {'class': format_class_name(CLASS_NAMES[i]),
             'confidence': round(float(probs[i]) * 100, 2)}
            for i in top5_idx
        ]

        info = get_disease_info(top_class)
        return jsonify({
            'success':     True,
            'predicted':   format_class_name(top_class),
            'raw_class':   top_class,
            'crop':        get_crop_name(top_class),
            'confidence':  round(top_conf, 2),
            'top5':        top5,
            'severity':    info['severity'],
            'description': info['description'],
            'treatment':   info['treatment'],
            'color':       info['color'],
            'image_b64':   img_b64,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'device':       str(device),
        'num_classes':  NUM_CLASSES,
        'classes':      CLASS_NAMES,
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Cucurbit Leaf Disease Detection System")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)