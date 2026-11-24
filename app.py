from flask import Flask, render_template, request
import torch
from torchvision.models import mobilenet_v2
from utils import preprocess, predict, generate_gradcam
import os
import cv2

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("models", "mobilenetv2_best.pt")

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

CONF_THRESHOLD = 0.65

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probs = None
    confidence = None
    orig_img = None
    cam_img = None

    if request.method == "POST":
        file = request.files["file"]
        orig_path = os.path.join("static", "original.png")
        cam_path = os.path.join("static", "gradcam.png")
        file.save(orig_path)

        img_tensor, pil_img = preprocess(orig_path)
        img_tensor = img_tensor.to(device)

        label, probs, confidence = predict(model, img_tensor)

        if confidence < CONF_THRESHOLD:
            result = "Uncertain – Needs Clinical Review"
        else:
            result = label.capitalize()

        class_idx = probs.index(max(probs))
        cam = generate_gradcam(model, img_tensor, class_idx, pil_img)
        cv2.imwrite(cam_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

        orig_img = "static/original.png"
        cam_img = "static/gradcam.png"

    return render_template(
        "index.html",
        result=result,
        probs=probs,
        confidence=confidence,
        orig_img=orig_img,
        cam_img=cam_img
    )

if __name__ == "__main__":
    app.run(debug=True)
