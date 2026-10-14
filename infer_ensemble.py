"""
Rock sample ensemble inference: runs ResNet_18, EfficientNet_B0, MobileNetV4_Small,
ShuffleNet_V2, and YOLO_v11_cls on one image, then averages their softmax
probabilities to get a final prediction.

Usage: python infer_ensemble.py path/to/rock_image.jpg
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import timm
from ultralytics import YOLO

NUM_CLASSES = 47
LOW_CONF_THRESHOLD = 50.0  # percent; below this the image shows a warning


def find_models_dir():
    """Look for the .pth files next to this script (Models/ or Models/Models/),
    then fall back to the old D:/Models/Models location."""
    here = Path(__file__).resolve().parent
    candidates = [here / "Models" / "Models", here / "Models", here, Path("D:/Models/Models")]
    for c in candidates:
        if (c / "ResNet_18.pth").exists():
            return str(c)
    raise FileNotFoundError(
        "ResNet_18.pth not found. Checked: " + ", ".join(str(c) for c in candidates)
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
        x = x.mean(dim=[2, 3])  # manual global average pool -> (B, 960)
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
    result = model.load_state_dict(state_dict, strict=False)
    # MobileNetV4 conv_head/norm_head keys are expected to be "unexpected" (bypassed on purpose).
    missing = list(result.missing_keys)
    unexpected = [k for k in result.unexpected_keys
                  if not (cls is MobileNetV4Classifier and ("conv_head" in k or "norm_head" in k))]
    if missing or unexpected:
        print(f"  [WARNING {weight_file}] missing: {missing[:5]}  unexpected: {unexpected[:5]}")
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Class index -> rock name (confirmed: 47 training folders in A-Z order).
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Amphibolite", 1: "Andesite", 2: "Anthracite", 3: "Basalt",
    4: "Blueschist", 5: "Carbonate", 6: "Carbonatite", 7: "Chert",
    8: "Coal", 9: "Coarse", 10: "Diamictite", 11: "Dolomite",
    12: "Eclogite", 13: "Evaporite", 14: "Flint", 15: "Gabbro",
    16: "Gneissic", 17: "Granite", 18: "Granulite", 19: "Greenschist",
    20: "Greywacke", 21: "Hornfels", 22: "Komatiite", 23: "Marble",
    24: "Mudstone_Shale", 25: "Obsidian", 26: "Oolite", 27: "Pegmatite",
    28: "Phyllite", 29: "Porphyry", 30: "Pyroclastic", 31: "Pyroxenite",
    32: "Quartz", 33: "Quartzite", 34: "Quartzolite", 35: "Rhyolite",
    36: "Sandstone", 37: "Scoria", 38: "Serpentinite", 39: "Shale",
    40: "Siltstone", 41: "Slate", 42: "Talc", 43: "Tephrite",
    44: "Travertine", 45: "Turbidite", 46: "Wackestone",
}


def get_font(size):
    for font_name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_image(img, top3):
    """Write predicted class, rock name, confidence and runner-ups in the top-right corner.
    top3 = [(class_idx, confidence_percent), ...]"""
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    W, H = out.size

    class_idx, confidence = top3[0]
    rock_name = CLASS_NAMES.get(class_idx, "Name not set")

    # (text, color) pairs
    lines = [
        (f"Class: {class_idx}", (255, 255, 255, 255)),
        (f"Rock: {rock_name}", (255, 255, 255, 255)),
        (f"Confidence: {confidence:.1f}%", (255, 255, 255, 255)),
    ]
    if confidence < LOW_CONF_THRESHOLD:
        lines.append(("LOW CONFIDENCE", (255, 90, 90, 255)))
    for rank, (i, c) in enumerate(top3[1:], start=2):
        lines.append((f"#{rank}: {CLASS_NAMES.get(i, 'Name not set')} ({c:.1f}%)", (220, 220, 220, 255)))

    pad = max(6, W // 60)
    font_size = max(14, W // 28)
    while True:  # shrink the font if the box would not fit inside the image
        font = get_font(font_size)
        boxes = [draw.textbbox((0, 0), t, font=font) for t, _ in lines]
        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        gap = pad // 2
        box_w = max(widths) + 2 * pad
        box_h = sum(heights) + gap * (len(lines) - 1) + 2 * pad
        if (box_w <= W - 2 * pad and box_h <= H - 2 * pad) or font_size <= 10:
            break
        font_size -= 2

    x1 = max(0, W - box_w - pad)
    y1 = pad
    draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=(0, 0, 0, 175))

    y = y1 + pad
    for (t, color), b, h in zip(lines, boxes, heights):
        draw.text((x1 + pad, y - b[1]), t, font=font, fill=color)
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
        print(f"  {name:20s} -> class {top_idx:2d} {CLASS_NAMES.get(top_idx, '?'):16s} ({top_conf:.2f}%)")

    # Ensemble: average softmax probabilities across all 5 models
    avg_probs = torch.stack(all_probs).mean(dim=0)
    top5_prob, top5_idx = torch.topk(avg_probs, 5)

    print("\nEnsemble (averaged) top-5 prediction:")
    for rank, (idx, p) in enumerate(zip(top5_idx.tolist(), top5_prob.tolist()), start=1):
        print(f"  {rank}. class {idx:2d} {CLASS_NAMES.get(idx, '?'):16s} —  {p*100:.2f}%")

    # Final prediction = ensemble top-1; write top-3 onto a copy of the test image
    final_idx = int(top5_idx[0])
    final_conf = float(top5_prob[0]) * 100
    final_name = CLASS_NAMES.get(final_idx, "Name not set")
    print(f"\nFINAL: class {final_idx} | rock: {final_name} | confidence: {final_conf:.2f}%")
    if final_conf < LOW_CONF_THRESHOLD:
        print("NOTE: low confidence - the models disagree, treat this result with caution.")

    top3 = [(i, p * 100) for i, p in zip(top5_idx.tolist()[:3], top5_prob.tolist()[:3])]
    annotated = annotate_image(img, top3)

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