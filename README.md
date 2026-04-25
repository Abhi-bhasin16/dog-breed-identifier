---
title: Dog Breed Identifier 🐶
emoji: 🐶
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# Dog Breed Identifier

A dog breed classifier trained on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) — **120 breeds**, fine-tuned **EfficientNet-B2** with PyTorch.

## How to use

Upload any photo of a dog and the model returns the top 5 predicted breeds with confidence scores.

## Local setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (requires the archive/ dataset folder)
python train.py

# 3. Run the app locally
python app.py
```

## Deployment on Hugging Face Spaces

After training:

1. **Create a model repo** on [huggingface.co](https://huggingface.co) and upload `model.pth`:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload Abhibhasin16/dog-breed-classifier model.pth model.pth
   ```

2. Push code to the Space:
   ```bash
   git push https://Abhibhasin16:YOUR_TOKEN@huggingface.co/spaces/Abhibhasin16/dog-breed-identifier main
   ```

## Model details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B2 |
| Dataset | Stanford Dogs (120 breeds) |
| Input size | 260×260 |
| Training | 2-phase: head warmup → full fine-tune |
| Augmentation | RandomResizedCrop, HorizontalFlip, ColorJitter, Rotation |

## Breeds supported

120 breeds including Chihuahua, Labrador, Golden Retriever, German Shepherd, Poodle, Beagle, and many more.
