import random
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONFIG
# ============================================================

SEED = 42
NUM_IMAGES = 5
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EYE_TEST_DIR = PROJECT_ROOT / "data" / "eye_face" / "combined" / "test"
SKIN_TEST_DIR = PROJECT_ROOT / "data" / "skin_coat" / "combined_clean" / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qualitative_all_models_layercam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

RGB_MEAN, RGB_STD = IMAGENET_MEAN, IMAGENET_STD
HSV_MEAN = HSV_STD = YCBCR_MEAN = YCBCR_STD = None  # set below
HSV_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
HSV_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
YCBCR_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
YCBCR_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


# ============================================================
# COLOR SPACES
# ============================================================

def rgb_to_hsv(rgb):
    rgb = rgb.clamp(0, 1)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    max_val = torch.max(rgb, dim=1).values
    min_val = torch.min(rgb, dim=1).values
    delta = max_val - min_val

    hue = torch.zeros_like(max_val)
    mask = delta > 1e-6
    rmask = mask & (max_val == r)
    gmask = mask & (max_val == g)
    bmask = mask & (max_val == b)

    hue[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6
    hue[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2
    hue[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4
    hue = hue / 6.0

    saturation = torch.zeros_like(max_val)
    max_mask = max_val > 1e-6
    saturation[max_mask] = delta[max_mask] / max_val[max_mask]

    return torch.stack([hue, saturation, max_val], dim=1)


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=1).clamp(0, 1)


def sobel_texture(rgb):
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=rgb.dtype, device=rgb.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=rgb.dtype, device=rgb.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


# ============================================================
# LAYERCAM
# ============================================================

def _cam_from(activation, gradient):
    cam = (torch.relu(gradient) * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(
        cam, size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False
    )[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach()


class LayerCAM:
    """Single target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, score):
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.activations is None:
            raise RuntimeError("LayerCAM activation was not captured.")
        if self.gradients is None:
            raise RuntimeError("LayerCAM gradient was not captured "
                               "(output does not require grad).")

        return _cam_from(self.activations, self.gradients).cpu().numpy()

    def close(self):
        self.hook.remove()


class MultiBranchLayerCAM:
    """One target layer per branch; branch CAMs are averaged."""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = []
        self.gradients = {}
        self.hooks = [
            layer.register_forward_hook(self._forward_hook)
            for layer in target_layers
        ]

    def _forward_hook(self, module, inputs, output):
        index = len(self.activations)
        self.activations.append(output)

        if isinstance(output, torch.Tensor) and output.requires_grad:
            def save_gradient(gradient, index=index):
                self.gradients[index] = gradient
            output.register_hook(save_gradient)

    def generate(self, logits, class_idx=None):
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        self.gradients = {}
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        n = len(self.target_layers)
        if len(self.activations) != n:
            raise RuntimeError(
                f"Activation count mismatch: expected {n}, "
                f"got {len(self.activations)}."
            )

        cams = []
        for i, activation in enumerate(self.activations):
            if i not in self.gradients:
                raise RuntimeError(f"Missing gradient for branch {i}.")
            cams.append(_cam_from(activation, self.gradients[i]))

        combined = torch.stack(cams, dim=0).mean(dim=0).clamp(0, 1)
        return combined.cpu().numpy()

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_state_dict(path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


# ============================================================
# EYE BASELINES
# ============================================================

def build_eye_efficientnet(num_classes=9):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_eye_densenet(num_classes=9):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


# ============================================================
# CSF-NET
# ============================================================

class MobileNetFeatureExtractor(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        model = models.mobilenet_v3_small(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.projection = model.classifier[0]
        self.output_dim = 1024

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class CSFNet(nn.Module):

    def __init__(self, num_classes=9):
        super().__init__()

        # pretrained=False: weights come from the checkpoint anyway,
        # so no download is needed.
        self.rgb_branch = MobileNetFeatureExtractor(pretrained=False)
        self.hsv_branch = MobileNetFeatureExtractor(pretrained=False)
        self.ycbcr_branch = MobileNetFeatureExtractor(pretrained=False)

        feature_dim = 1024
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=1, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, hsv, ycbcr):
        tokens = torch.stack(
            [self.rgb_branch(rgb), self.hsv_branch(hsv), self.ycbcr_branch(ycbcr)],
            dim=1
        )
        attended, _ = self.attention(tokens, tokens, tokens)
        attended = self.norm(attended + tokens)
        return self.classifier(attended.mean(dim=1))


# ============================================================
# MCFA-NET V2
# ============================================================

def normalize_with_stats(x, mean, std):
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def make_efficientnet():
    # weights=None: the checkpoint supplies all weights
    # (also fixes the undefined FREEZE_EARLY_BLOCKS name).
    backbone = models.efficientnet_b0(weights=None)
    feature_dim = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


def _projection(feature_dim, token_dim, dropout):
    return nn.Sequential(
        nn.Linear(feature_dim, token_dim),
        nn.LayerNorm(token_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class MCFA_Net_V2(nn.Module):

    def __init__(self, num_classes, token_dim=128, num_heads=4,
                 num_layers=2, dropout=0.25):
        super().__init__()

        self.rgb_backbone, feature_dim = make_efficientnet()
        self.hsv_backbone, _ = make_efficientnet()
        self.ycbcr_backbone, _ = make_efficientnet()

        self.rgb_projection = _projection(feature_dim, token_dim, dropout)
        self.hsv_projection = _projection(feature_dim, token_dim, dropout)
        self.ycbcr_projection = _projection(feature_dim, token_dim, dropout)

        self.branch_gate = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(token_dim, 3),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=token_dim * 4, dropout=0.15,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.cross_color_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fusion_score = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )
        self.rgb_residual = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(token_dim)
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_branch_weights=False):
        rgb = x
        hsv = rgb_to_hsv(x)
        ycbcr = rgb_to_ycbcr(x)

        rgb = normalize_with_stats(rgb, RGB_MEAN, RGB_STD)
        hsv = normalize_with_stats(hsv, HSV_MEAN, HSV_STD)
        ycbcr = normalize_with_stats(ycbcr, YCBCR_MEAN, YCBCR_STD)

        rgb_token = self.rgb_projection(self.rgb_backbone(rgb))
        hsv_token = self.hsv_projection(self.hsv_backbone(hsv))
        ycbcr_token = self.ycbcr_projection(self.ycbcr_backbone(ycbcr))

        tokens = torch.stack([rgb_token, hsv_token, ycbcr_token], dim=1)

        combined = torch.cat([rgb_token, hsv_token, ycbcr_token], dim=1)
        branch_weights = torch.softmax(self.branch_gate(combined), dim=1)
        gated_tokens = tokens * branch_weights.unsqueeze(-1)

        attended = self.cross_color_transformer(gated_tokens)

        fusion_weights = torch.softmax(self.fusion_score(attended), dim=1)
        fused_color = (attended * fusion_weights).sum(dim=1)

        rgb_residual = self.rgb_residual(rgb_token)
        residual_gate = self.residual_gate(
            torch.cat([fused_color, rgb_residual], dim=1)
        )
        fused = residual_gate * fused_color + (1.0 - residual_gate) * rgb_residual
        fused = self.fusion_norm(fused)

        logits = self.classifier(fused)

        if return_branch_weights:
            return logits, {
                "branch_weights": branch_weights,
                "fusion_weights": fusion_weights.squeeze(-1),
                "residual_gate_mean": residual_gate.mean(dim=1),
            }
        return logits


# ============================================================
# CSTF-NET
# ============================================================

class DepthwiseSeparableBlock(nn.Module):
    """
    Parameter names match the trained checkpoint:
        depthwise.weight / pointwise.weight / bn.*
    bn_after_pointwise is auto-detected from the checkpoint (see
    build_skin_cstf): True  -> depthwise -> pointwise -> bn -> act
                      False -> depthwise -> bn -> act -> pointwise
    """

    def __init__(self, in_channels, out_channels, bn_after_pointwise=True):
        super().__init__()
        self.bn_after_pointwise = bn_after_pointwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=2, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(
            out_channels if bn_after_pointwise else in_channels
        )
        self.act = nn.GELU()

    def forward(self, x):
        if self.bn_after_pointwise:
            return self.act(self.bn(self.pointwise(self.depthwise(x))))
        return self.pointwise(self.act(self.bn(self.depthwise(x))))


class CSTFBranch(nn.Module):

    def __init__(self, input_channels, bn_after_pointwise=True):
        super().__init__()
        b = bn_after_pointwise
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            DepthwiseSeparableBlock(32, 64, b),
            DepthwiseSeparableBlock(64, 128, b),
            DepthwiseSeparableBlock(128, 256, b),
            DepthwiseSeparableBlock(256, 256, b),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.projection(self.features(x))


class CSTFNet(nn.Module):

    def __init__(self, num_classes=11, bn_after_pointwise=True):
        super().__init__()

        # Named rgb_branch (not rgb_backbone) to match the checkpoint.
        self.rgb_branch = models.efficientnet_b0(weights=None)
        rgb_dim = self.rgb_branch.classifier[1].in_features
        self.rgb_branch.classifier = nn.Identity()

        self.rgb_projection = nn.Sequential(
            nn.Linear(rgb_dim, 128), nn.GELU(), nn.Dropout(0.15)
        )
        self.color_branch = CSTFBranch(3, bn_after_pointwise)
        self.texture_branch = CSTFBranch(1, bn_after_pointwise)

        self.branch_gate = nn.Sequential(
            nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 3)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.15,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, color, texture):
        rgb_feat = self.rgb_projection(self.rgb_branch(rgb))
        color_feat = self.color_branch(color)
        texture_feat = self.texture_branch(texture)

        tokens = torch.stack([rgb_feat, color_feat, texture_feat], dim=1)
        combined = torch.cat([rgb_feat, color_feat, texture_feat], dim=1)

        weights = torch.softmax(self.branch_gate(combined), dim=1)
        weighted_tokens = tokens * weights.unsqueeze(-1)

        transformed = self.transformer(weighted_tokens)
        transformed_global = transformed.mean(dim=1)
        original_global = weighted_tokens.mean(dim=1)

        gate = self.fusion_gate(
            torch.cat([original_global, transformed_global], dim=1)
        )
        fused = gate * transformed_global + (1 - gate) * original_global
        return self.classifier(fused)


def build_skin_cstf(state):
    """Needs the checkpoint to decide where the BatchNorm sits."""
    bn_channels = state["color_branch.features.3.bn.weight"].shape[0]
    # first block is 32 -> 64 channels
    return CSTFNet(11, bn_after_pointwise=(bn_channels == 64))


build_skin_cstf.needs_state = True


# ============================================================
# SKIN BASELINES
# ============================================================

def build_skin_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 11)
    return model


def build_skin_efficientnet_v2s():
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 11)
    return model


def build_skin_convnext():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 11)
    return model


# ============================================================
# MODEL REGISTRY  (column order in the figure = dict order)
# ============================================================

EYE_MODELS = {
    "EfficientNet-B0": (
        build_eye_efficientnet,
        PROJECT_ROOT / "outputs" / "eye_combined" / "best_model.pth"),
    "DenseNet-121": (
        build_eye_densenet,
        PROJECT_ROOT / "outputs" / "eye_densenet121" / "best_model.pth"),
    "CSF-Net": (
        lambda: CSFNet(9),
        PROJECT_ROOT / "outputs" / "eye_csfnet" / "best_csfnet.pth"),
    "MCFA-Net V2": (
        lambda: MCFA_Net_V2(9),
        PROJECT_ROOT / "outputs" / "eye_mcfa_v2" / "best_model.pth"),
}

SKIN_MODELS = {
    "EfficientNet-B0": (
        build_skin_efficientnet,
        PROJECT_ROOT / "outputs" / "skin_efficientnet" / "best_model.pth"),
    "EfficientNetV2-S": (
        build_skin_efficientnet_v2s,
        PROJECT_ROOT / "outputs" / "skin_efficientnet_v2s" / "best_model.pth"),
    "ConvNeXt-Tiny": (
        build_skin_convnext,
        PROJECT_ROOT / "outputs" / "skin_convnext_tiny" / "best_model.pth"),
    "CSTF-Net": (
        build_skin_cstf,
        PROJECT_ROOT / "outputs" / "skin_cstf" / "best_model.pth"),
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(builder, checkpoint_path):
    if not checkpoint_path.exists():
        outputs = PROJECT_ROOT / "outputs"
        found = sorted(
            str(p.relative_to(PROJECT_ROOT)) for p in outputs.rglob("*.pth")
        ) if outputs.exists() else []
        hint = "\n  ".join(found) if found else "(no .pth files under outputs/)"
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available checkpoints:\n  {hint}"
        )

    state = load_state_dict(checkpoint_path)
    model = builder(state) if getattr(builder, "needs_state", False) else builder()
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    # Re-enable grads in case early layers were frozen at training time;
    # otherwise no gradient reaches the hooked layer (inference only).
    for p in model.parameters():
        p.requires_grad_(True)

    return model


# ============================================================
# TARGET LAYERS
# ============================================================

def get_target_layer(model, model_name, modality):

    if modality == "eye":
        if model_name == "EfficientNet-B0":
            return model.features[-1][0]
        if model_name == "DenseNet-121":
            # Whole last dense block (1024 ch). The old target
            # (denselayer16.conv2) only has 32 growth channels.
            return model.features.denseblock4
        if model_name == "CSF-Net":
            return [
                model.rgb_branch.features[-1],
                model.hsv_branch.features[-1],
                model.ycbcr_branch.features[-1],
            ]
        if model_name == "MCFA-Net V2":
            return [
                model.rgb_backbone.features[-1][0],
                model.hsv_backbone.features[-1][0],
                model.ycbcr_backbone.features[-1][0],
            ]

    else:
        if model_name in ("EfficientNet-B0", "EfficientNetV2-S"):
            return model.features[-1][0]
        if model_name == "ConvNeXt-Tiny":
            # features[-1] is a Sequential of CNBlocks, so index the last
            # block first (the old model.features[-1].block[0] crashes).
            return model.features[-1][-1].block[0]
        if model_name == "CSTF-Net":
            return [
                model.rgb_branch.features[-1][0],
                model.color_branch.features[-2],
                model.texture_branch.features[-2],
            ]

    raise ValueError(f"Unknown model: {model_name} ({modality})")


# ============================================================
# INPUT / FORWARD
# ============================================================

def prepare_rgb(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def imagenet_normalize(tensor):
    return (tensor - IMAGENET_MEAN.to(tensor.device)) / IMAGENET_STD.to(tensor.device)


def forward_model(model, model_name, modality, rgb):
    rgb_input = imagenet_normalize(rgb)

    if modality == "eye":
        if model_name in ("EfficientNet-B0", "DenseNet-121"):
            return model(rgb_input)
        if model_name == "CSF-Net":
            return model(rgb_input, rgb_to_hsv(rgb), rgb_to_ycbcr(rgb))
        if model_name == "MCFA-Net V2":
            return model(rgb)  # does its own colour conversion/normalisation
    else:
        if model_name in ("EfficientNet-B0", "EfficientNetV2-S", "ConvNeXt-Tiny"):
            return model(rgb_input)
        if model_name == "CSTF-Net":
            return model(rgb_input, rgb_to_hsv(rgb), sobel_texture(rgb))

    raise ValueError(model_name)


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(rgb_image, cam, alpha=0.45):
    original = cv2.resize(
        np.asarray(rgb_image.convert("RGB")), (IMAGE_SIZE, IMAGE_SIZE)
    )
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original, 1.0 - alpha, heatmap, alpha, 0)


# ============================================================
# IMAGE SELECTION
# ============================================================

def get_class_names(root):
    # same ordering torchvision.ImageFolder uses
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def select_images(root, count=5):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(root.rglob(ext))
    files = sorted(set(files))

    if len(files) < count:
        raise RuntimeError(f"Only {len(files)} images found in {root}")

    return random.sample(files, count)


# ============================================================
# COMPUTE CAMS: one model loaded at a time, all images per model
# ============================================================

def compute_overlays(image_paths, models_dict, modality, class_names=None):
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    rgbs = [prepare_rgb(im) for im in pil_images]
    n = len(pil_images)

    overlays = {}

    for model_name, (builder, checkpoint) in models_dict.items():
        print(f"  [{modality}] {model_name}")

        try:
            model = load_model(builder, checkpoint)
        except Exception:
            print(f"    could not load {model_name}:")
            traceback.print_exc()
            overlays[model_name] = [None] * n
            continue

        per_image = []

        for i, (image, rgb) in enumerate(zip(pil_images, rgbs)):
            generator = None
            try:
                target = get_target_layer(model, model_name, modality)
                multi = isinstance(target, list)
                generator = (
                    MultiBranchLayerCAM(model, target) if multi
                    else LayerCAM(model, target)
                )

                model.zero_grad(set_to_none=True)
                logits = forward_model(model, model_name, modality, rgb)
                pred = int(logits.argmax(dim=1).item())

                if multi:
                    cam = generator.generate(logits, class_idx=pred)
                else:
                    cam = generator.generate(logits[0, pred])

                per_image.append(make_overlay(image, cam))
                pname = (class_names[pred] if class_names and pred < len(class_names)
                         else "?")
                print(f"    image {i + 1}: pred {pred} ({pname}) | "
                      f"true: {image_paths[i].parent.name}")

            except Exception:
                print(f"    LayerCAM FAILED for {model_name}, image {i + 1}:")
                traceback.print_exc()
                per_image.append(None)

            finally:
                if generator is not None:
                    generator.close()

        overlays[model_name] = per_image

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    originals = [
        np.asarray(im.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB"))
        for im in pil_images
    ]
    return originals, overlays


# ============================================================
# TABLE-STYLE FIGURE  (header row + image rows + (a),(b).. captions)
# ============================================================

def _frame(ax, lw=1.2):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color("black")


def build_grid_figure(originals, overlays, save_path, cell_in=2.3):
    model_names = list(overlays.keys())
    n_rows = len(originals)
    n_cols = 1 + len(model_names)

    header_h, img_h, cap_h = 0.55, 1.0, 0.22
    ratios = [header_h]
    for _ in range(n_rows):
        ratios += [img_h, cap_h]

    fig_w = cell_in * n_cols
    fig_h = cell_in * (header_h + n_rows * img_h + n_rows * cap_h)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        len(ratios), n_cols, height_ratios=ratios,
        hspace=0, wspace=0, left=0.005, right=0.995, top=0.995, bottom=0.005
    )

    # header
    headers = ["INPUT\nIMAGES"] + model_names
    for c, text in enumerate(headers):
        ax = fig.add_subplot(gs[0, c])
        _frame(ax)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=13, fontweight="bold", transform=ax.transAxes)

    # rows
    for r in range(n_rows):
        img_row = 1 + 2 * r
        cap_row = img_row + 1

        ax = fig.add_subplot(gs[img_row, 0])
        ax.imshow(originals[r])
        _frame(ax)

        for c, name in enumerate(model_names, start=1):
            ax = fig.add_subplot(gs[img_row, c])
            overlay = overlays[name][r]
            if overlay is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
            else:
                ax.imshow(overlay)
            _frame(ax)

        cap = fig.add_subplot(gs[cap_row, :])
        _frame(cap)
        cap.text(0.5, 0.45, f"({chr(ord('a') + r)})", ha="center",
                 va="center", fontsize=12, fontweight="bold",
                 transform=cap.transAxes)

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  SAVED: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ALL-MODEL LAYERCAM QUALITATIVE GRIDS")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    eye_images = select_images(EYE_TEST_DIR, NUM_IMAGES)
    skin_images = select_images(SKIN_TEST_DIR, NUM_IMAGES)

    print("\nEye images:")
    for p in eye_images:
        print(f"  {p}")
    print("\nSkin images:")
    for p in skin_images:
        print(f"  {p}")

    print("\nEYE MODELS")
    originals, overlays = compute_overlays(
        eye_images, EYE_MODELS, "eye", get_class_names(EYE_TEST_DIR))
    build_grid_figure(originals, overlays,
                      OUTPUT_DIR / "eye_layercam_comparison.png")

    print("\nSKIN MODELS")
    originals, overlays = compute_overlays(
        skin_images, SKIN_MODELS, "skin", get_class_names(SKIN_TEST_DIR))
    build_grid_figure(originals, overlays,
                      OUTPUT_DIR / "skin_layercam_comparison.png")

    print(f"\nDONE. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()