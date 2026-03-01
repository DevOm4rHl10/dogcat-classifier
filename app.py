from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from classifier import DogCatCNN  # import your CNN class
import numpy as np
import cv2
from io import BytesIO
import base64
app = Flask(__name__)






# Grad-CAM implementation
def generate_gradcam(model, img_tensor, target_layer):
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on the target layer
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = torch.sigmoid(output)

    # Backward pass for the predicted class
    model.zero_grad()
    pred_class.backward()

    # Get hooks outputs
    grads = gradients[0].cpu().data.numpy()[0]          # C x H x W
    acts = activations[0].cpu().data.numpy()[0]        # C x H x W

    weights = np.mean(grads, axis=(1, 2))             # Global average pooling
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return cam
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DogCatCNN().to(device)
model.load_state_dict(torch.load("saved_models/dogcat_cnn.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output)
        pred = "Dog" if prob.item() > 0.5 else "Cat"
    return pred

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")
    
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        pred = "Dog" if prob.item() > 0.5 else "Cat"

    # Grad-CAM
    target_layer = model.conv3  # last conv layer
    cam = generate_gradcam(model, img_tensor, target_layer)

    # Overlay CAM on original image
    img_resized = np.array(img.resize((128,128)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * heatmap + 0.5 * img_resized)

    # Convert overlay to base64 for HTML
    overlay_pil = Image.fromarray(overlay)
    buffered = BytesIO()
    overlay_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template("index.html", prediction=pred, gradcam_image=img_str)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
