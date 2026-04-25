"""
Gradio web app for dog breed identification.
Loads model from Hugging Face Hub or a local model.pth file.

Set HF_MODEL_REPO env var to your HF model repo, e.g.:
    HF_MODEL_REPO=your-username/dog-breed-classifier
"""

import json
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Abhibhasin16/dog-breed-classifier")
LOCAL_CHECKPOINT = Path("model.pth")
LOCAL_CLASSES = Path("class_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize(280),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model loading ─────────────────────────────────────────────────────────────
def _build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model():
    checkpoint_path = LOCAL_CHECKPOINT

    if not checkpoint_path.exists() and HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from {HF_MODEL_REPO}...")
        checkpoint_path = Path(hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pth"))

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No model found. Either place model.pth here or set HF_MODEL_REPO env var."
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model = _build_model(len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, class_names


print("Loading model...")
model, CLASS_NAMES = load_model()
print(f"Model loaded — {len(CLASS_NAMES)} breeds")


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image: Image.Image):
    if image is None:
        return {}
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top5 = probs.topk(5)
    return {CLASS_NAMES[i]: float(p) for i, p in zip(top5.indices, top5.values)}


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES = []
example_dir = Path("examples")
if example_dir.exists():
    EXAMPLES = [[str(p)] for p in sorted(example_dir.glob("*.jpg"))[:6]]

with gr.Blocks(title="Dog Breed Identifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐶 Dog Breed Identifier
        Upload a photo of a dog and the model will identify its breed.
        Trained on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
        — **120 breeds**, fine-tuned EfficientNet-B2.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload a dog photo")
            submit_btn = gr.Button("Identify Breed", variant="primary")
        with gr.Column(scale=1):
            label_out = gr.Label(num_top_classes=5, label="Top 5 Predicted Breeds")

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=img_input, fn=predict, outputs=label_out)

    submit_btn.click(fn=predict, inputs=img_input, outputs=label_out)
    img_input.change(fn=predict, inputs=img_input, outputs=label_out)

    gr.Markdown(
        """
        ---
        *Model: EfficientNet-B2 · Dataset: Stanford Dogs (120 breeds)*
        """
    )

if __name__ == "__main__":
    demo.launch()
