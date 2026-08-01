"""
Vibecoded Detector -- local web app.

Loads your trained ResNet18 model once at startup and serves a single-page
UI where you can drop in one or more screenshots and get back the
probability that each one is "vibecoded", plus the average across all of them.

SETUP
-----
1. Put your trained model files here:
       models/final_model_state_dict.pt
       models/metadata.json
   (both were produced by the training notebook's final "save and download" step)

2. Install dependencies:
       pip install -r requirements.txt

3. Run:
       python app.py

4. Open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import json

from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(APP_DIR), "models")
MODEL_STATE_PATH = os.path.join(MODEL_DIR, "final_model_state_dict.pt")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

MAX_UPLOAD_MB = 25
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Load metadata + model once at startup (not per-request -- that would be slow)
# ---------------------------------------------------------------------------
if not os.path.exists(METADATA_PATH) or not os.path.exists(MODEL_STATE_PATH):
    raise FileNotFoundError(
        "Model files not found. Expected:\n"
        f"  {MODEL_STATE_PATH}\n"
        f"  {METADATA_PATH}\n"
        "Copy your trained model's final_model_state_dict.pt and metadata.json "
        "into the 'models/' folder."
    )

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)

IMAGE_SIZE = METADATA["image_size"]
NORM_MEAN = METADATA["normalize_mean"]
NORM_STD = METADATA["normalize_std"]
CLASS_NAMES = METADATA["class_names"]                        # ["not_vibecoded", "vibecoded"]
THRESHOLD = METADATA.get("classification_threshold", 0.5)

print(f"  image_size={IMAGE_SIZE}  threshold={THRESHOLD}  classes={CLASS_NAMES}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Building model and loading trained weights...")
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, METADATA["num_classes"])
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model ready.")

# ---------------------------------------------------------------------------
# Preprocessing -- must match training preprocessing exactly:
# resize preserving aspect ratio, pad (letterbox) to a square with white
# fill, then normalize with the same ImageNet mean/std used in training.
# ---------------------------------------------------------------------------
def resize_letterbox(img, target_size=IMAGE_SIZE, fill=(255, 255, 255)):
    img = img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


to_tensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def predict_single_image(pil_image):
    """Returns P(vibecoded) as a float in [0, 1]."""
    letterboxed = resize_letterbox(pil_image)
    tensor = to_tensor_and_normalize(letterboxed).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        prob_vibecoded = torch.softmax(output, dim=1)[0, 1].item()
    return prob_vibecoded


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", threshold=THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images were uploaded."}), 400

    results = []
    for f in files:
        if not allowed_file(f.filename):
            results.append({"filename": f.filename, "error": "Unsupported file type."})
            continue
        try:
            image_bytes = f.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            prob = predict_single_image(pil_image)
            results.append({
                "filename": f.filename,
                "probability": round(prob, 4),
                "label": CLASS_NAMES[1] if prob >= THRESHOLD else CLASS_NAMES[0],
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": f"Could not process image ({e})."})

    valid_probs = [r["probability"] for r in results if "probability" in r]

    average_probability = round(sum(valid_probs) / len(valid_probs), 4) if valid_probs else None
    average_label = None
    if average_probability is not None:
        average_label = CLASS_NAMES[1] if average_probability >= THRESHOLD else CLASS_NAMES[0]

    return jsonify({
        "results": results,
        "average_probability": average_probability,
        "average_label": average_label,
        "threshold": THRESHOLD,
        "class_names": CLASS_NAMES,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
