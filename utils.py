import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

CLASSES = ["benign", "malignant", "normal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor, img

def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, 0)

    return CLASSES[idx], probs.cpu().numpy().tolist(), float(confidence)

# -------- GRAD-CAM --------
def generate_gradcam(model, img_tensor, class_idx, original_img):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()

    weights = np.mean(grads, axis=(2, 3))[0]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[0, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, original_img.size)
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.array(original_img)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    h1.remove()
    h2.remove()

    return overlay
