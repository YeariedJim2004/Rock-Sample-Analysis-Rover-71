"""
Rock sample ensemble inference: runs ResNet_18, EfficientNet_B0, MobileNetV4_Small,
ShuffleNet_V2, and YOLO_v11_cls on one image, then averages their softmax
probabilities to get a final prediction.

Usage: python infer_ensemble.py path/to/rock_image.jpg
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import timm
from ultralytics import YOLO

NUM_CLASSES = 47
from pathlib import Path

def find_models_dir():
    """Look for the .pth files next to this script (Models/ or Models/Models/),
    then fall back to the old D:/Models/Models location."""
    here = Path(__file__).resolve().parent
    candidates = [here / "Models" / "Models", here / "Models", here, Path("D:/Models/Models")]
    for c in candidates:
        if (c / "ResNet_18.pth").exists():
            return str(c)
    raise FileNotFoundError(
        "ResNet_18.pth khuje pawa jay ni. Checked: " + ", ".join(str(c) for c in candidates)
    )

MODELS_DIR = find_models_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_head(in_dim, num_classes=NUM_CLASSES):
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )


# ---------- ResNet_18 ----------
class ResNet18Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = make_head(512)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).flatten(1)
        return self.classification_head(x)


# ---------- ShuffleNet_V2 ----------
class ShuffleNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.shufflenet_v2_x1_0(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = make_head(1024)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).flatten(1)
        return self.classification_head(x)


# ---------- EfficientNet_B0 ----------
class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.classification_head = make_head(1280)

    def forward(self, x):
        x = self.base_model(x)
        return self.classification_head(x)


# ---------- MobileNetV4_Small ----------
class MobileNetV4Classifier(nn.Module):
    """NOTE: base_model's conv_head/norm_head (960->1280 expansion) exist in the
    checkpoint but were NOT used during training -- the classification_head takes
    the raw 960-dim block output directly. So forward() must bypass conv_head and
    norm_head entirely, not call self.base_model(x) directly."""

    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=0)
        self.classification_head = make_head(960)

    def forward(self, x):
        x = self.base_model.conv_stem(x)
        x = self.base_model.bn1(x)
        x = self.base_model.blocks(x)
        x = x.mean(dim=[2, 3])  # manual global average pool -> (B, 960), skips conv_head/norm_head
        return self.classification_head(x)


# ---------- YOLO_v11_cls ----------
class YoloBackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        full = YOLO("yolo11n-cls.yaml").model
        self.model = full.model

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == 10:
                x = layer.conv(x)
                break
            x = layer(x)
        return x


class YoloClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.yolo_backbone = YoloBackboneWrapper()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = make_head(1280)

    def forward(self, x):
        x = self.yolo_backbone(x)
        x = self.pool(x).flatten(1)
        return self.classification_head(x)


MODEL_REGISTRY = {
    "ResNet_18": (ResNet18Classifier, "ResNet_18.pth"),
    "EfficientNet_B0": (EfficientNetClassifier, "EfficientNet_B0.pth"),
    "MobileNetV4_Small": (MobileNetV4Classifier, "MobileNetV4_Small.pth"),
    "ShuffleNet_V2": (ShuffleNetClassifier, "ShuffleNet_V2.pth"),
    "YOLO_v11_cls": (YoloClassifier, "YOLO_v11_cls.pth"),
}


def load_model(cls, weight_file):
    model = cls()
    state_dict = torch.load(f"{MODELS_DIR}/{weight_file}", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Class index -> rock name.
# The checkpoints only store class INDICES (0-46), not names, so the names
# must come from your original dataset/training label order.
# Fill in as many as you know, in the form:   index: "Rock Name",
# Any index left out is shown on the image as "Name not set".
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    # 0: "...",
    # 1: "...",
}


def get_font(size):
    for font_name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_image(img, class_idx, confidence):
    """Write the predicted class + rock name in the top-right corner."""
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    W, H = out.size

    rock_name = CLASS_NAMES.get(class_idx, "Name not set")
    lines = [f"Class: {class_idx}", f"Rock: {rock_name}", f"Confidence: {confidence:.1f}%"]

    pad = max(6, W // 60)
    font_size = max(14, W // 28)
    while True:  # shrink the font if the box would not fit inside the image
        font = get_font(font_size)
        boxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        box_w = max(widths) + 2 * pad
        if box_w <= W - 2 * pad or font_size <= 10:
            break
        font_size -= 2

    gap = pad // 2
    box_h = sum(heights) + gap * (len(lines) - 1) + 2 * pad
    x1 = max(0, W - box_w - pad)
    y1 = pad
    draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=(0, 0, 0, 175))

    y = y1 + pad
    for t, b, h in zip(lines, boxes, heights):
        draw.text((x1 + pad, y - b[1]), t, font=font, fill=(255, 255, 255, 255))
        y += h + gap
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_ensemble.py path/to/image.jpg")
        return

    image_path = sys.argv[1]
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    all_probs = []
    print(f"\nImage: {image_path}\n")
    print("Per-model predictions:")

    for name, (cls, weight_file) in MODEL_REGISTRY.items():
        model = load_model(cls, weight_file)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        all_probs.append(probs)

        top_idx = int(torch.argmax(probs))
        top_conf = float(probs[top_idx]) * 100
        print(f"  {name:20s} -> class {top_idx:2d}  ({top_conf:.2f}%)")

    # Ensemble: average softmax probabilities across all 5 models
    avg_probs = torch.stack(all_probs).mean(dim=0)
    top5_prob, top5_idx = torch.topk(avg_probs, 5)

    print("\nEnsemble (averaged) top-5 prediction:")
    for rank, (idx, p) in enumerate(zip(top5_idx.tolist(), top5_prob.tolist()), start=1):
        print(f"  {rank}. class {idx:2d}  —  {p*100:.2f}%")

    # Final prediction = ensemble top-1; write it onto a copy of the test image
    final_idx = int(top5_idx[0])
    final_conf = float(top5_prob[0]) * 100
    final_name = CLASS_NAMES.get(final_idx, "Name not set")
    print(f"\nFINAL: class {final_idx} | rock: {final_name} | confidence: {final_conf:.2f}%")

    annotated = annotate_image(img, final_idx, final_conf)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{Path(image_path).stem}_result.jpg"
    annotated.save(out_path, quality=95)
    print(f"Annotated image saved: {out_path}")
    try:
        os.startfile(out_path)  # opens the result image (Windows)
    except Exception:
        pass


if __name__ == "__main__":
    main()
