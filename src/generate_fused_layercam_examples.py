"""
generate_fused_layercam_examples.py

Generate qualitative Fused LayerCAM examples.

Eye:
    MCFA-Net V2
    RGB + HSV + YCbCr
    -> LayerCAM per branch
    -> learned branch-weight fusion
    -> one final Fused LayerCAM

Skin:
    CSTF-Net
    RGB + HSV + Texture
    -> LayerCAM per branch
    -> learned branch-weight fusion
    -> one final Fused LayerCAM

Output:
    outputs/
        qualitative_layercam/
            eye/
            skin/

Exactly 5 random test images are selected from each dataset.
"""

import os
import random
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EYE_TEST_DIR = (
    PROJECT_ROOT /
    "data" /
    "eye_face" /
    "combined" /
    "test"
)

SKIN_TEST_DIR = (
    PROJECT_ROOT /
    "data" /
    "skin_coat" /
    "combined_clean" /
    "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "qualitative_layercam"
)

EYE_CHECKPOINT = (
    PROJECT_ROOT /
    "outputs" /
    "eye_mcfa_v2" /
    "best_model.pth"
)

SKIN_CHECKPOINT = (
    PROJECT_ROOT /
    "outputs" /
    "skin_cstf" /
    "best_model.pth"
)

IMAGE_SIZE = 224
NUM_RANDOM_IMAGES = 5
SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# COLOR CONVERSIONS
# ============================================================

def rgb_to_hsv(x):
    """
    RGB -> HSV.

    x:
        [B,3,H,W]
        values in [0,1]
    """

    x = x.clamp(0.0, 1.0)

    r = x[:, 0]
    g = x[:, 1]
    b = x[:, 2]

    max_val = torch.max(
        x,
        dim=1
    ).values

    min_val = torch.min(
        x,
        dim=1
    ).values

    delta = max_val - min_val

    hue = torch.zeros_like(max_val)

    nonzero = delta > 1e-6

    mask = (
        nonzero
        & (max_val == r)
    )

    hue[mask] = (
        (
            g[mask]
            - b[mask]
        )
        / delta[mask]
    ) % 6.0

    mask = (
        nonzero
        & (max_val == g)
    )

    hue[mask] = (
        (
            b[mask]
            - r[mask]
        )
        / delta[mask]
    ) + 2.0

    mask = (
        nonzero
        & (max_val == b)
    )

    hue[mask] = (
        (
            r[mask]
            - g[mask]
        )
        / delta[mask]
    ) + 4.0

    hue = hue / 6.0

    saturation = torch.zeros_like(max_val)

    nonzero_max = max_val > 1e-6

    saturation[nonzero_max] = (
        delta[nonzero_max]
        / max_val[nonzero_max]
    )

    value = max_val

    return torch.stack(
        [
            hue,
            saturation,
            value,
        ],
        dim=1,
    )


def rgb_to_ycbcr(x):
    """
    RGB -> YCbCr.
    """

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )

    cb = (
        -0.168736 * r
        -0.331264 * g
        + 0.5 * b
        + 0.5
    )

    cr = (
        0.5 * r
        -0.418688 * g
        -0.081312 * b
        + 0.5
    )

    return torch.cat(
        [
            y,
            cb,
            cr,
        ],
        dim=1,
    ).clamp(
        0.0,
        1.0,
    )


def sobel_texture(x):
    """
    RGB -> Sobel gradient magnitude.
    """

    gray = (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )

    sobel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(
        1,
        1,
        3,
        3,
    )

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(
        1,
        1,
        3,
        3,
    )

    grad_x = F.conv2d(
        gray,
        sobel_x,
        padding=1,
    )

    grad_y = F.conv2d(
        gray,
        sobel_y,
        padding=1,
    )

    magnitude = torch.sqrt(
        grad_x ** 2
        + grad_y ** 2
        + 1e-6
    )

    maximum = magnitude.amax(
        dim=(2, 3),
        keepdim=True,
    )

    magnitude = magnitude / (
        maximum + 1e-6
    )

    return magnitude


# ============================================================
# NORMALIZATION
# ============================================================

RGB_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(
    1,
    3,
    1,
    1,
)

RGB_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(
    1,
    3,
    1,
    1,
)

HSV_MEAN = torch.tensor(
    [0.5, 0.5, 0.5]
).view(
    1,
    3,
    1,
    1,
)

HSV_STD = torch.tensor(
    [0.5, 0.5, 0.5]
).view(
    1,
    3,
    1,
    1,
)

YCBCR_MEAN = torch.tensor(
    [0.5, 0.5, 0.5]
).view(
    1,
    3,
    1,
    1,
)

YCBCR_STD = torch.tensor(
    [0.5, 0.5, 0.5]
).view(
    1,
    3,
    1,
    1,
)


def normalize(
    x,
    mean,
    std,
):

    mean = mean.to(
        device=x.device,
        dtype=x.dtype,
    )

    std = std.to(
        device=x.device,
        dtype=x.dtype,
    )

    return (
        x - mean
    ) / std


# ============================================================
# LAYERCAM
# ============================================================

class LayerCAM:
    """
    LayerCAM implementation.

    The hook must be installed before forward().
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

    def forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

        if (
            isinstance(
                output,
                torch.Tensor,
            )
            and output.requires_grad
        ):

            output.register_hook(
                self.gradient_hook
            )

    def gradient_hook(
        self,
        gradient,
    ):

        self.gradients = gradient

    def generate(
        self,
        logits,
        class_index,
        output_size,
    ):

        self.gradients = None

        self.model.zero_grad(
            set_to_none=True
        )

        score = logits[
            :,
            class_index
        ].sum()

        score.backward(
            retain_graph=True
        )

        if self.gradients is None:
            raise RuntimeError(
                "LayerCAM gradient was not captured."
            )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        # LayerCAM:
        # positive gradients weight each
        # spatial activation location.
        positive_gradients = F.relu(
            gradients
        )

        cam = (
            positive_gradients
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ]

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 1e-8:
            cam /= maximum

        return cam

    def remove(
        self
    ):

        if self.forward_handle:
            self.forward_handle.remove()

            self.forward_handle = None


# ============================================================
# MCFA-NET V2
# ============================================================

def build_efficientnet():

    model = models.efficientnet_b0(
        weights=None
    )

    feature_dim = (
        model.classifier[-1]
        .in_features
    )

    model.classifier = nn.Identity()

    return model, feature_dim


class MCFA_Net_V2(
    nn.Module
):

    def __init__(
        self,
        num_classes,
        token_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.25,
    ):

        super().__init__()

        self.rgb_backbone, feature_dim = (
            build_efficientnet()
        )

        self.hsv_backbone, _ = (
            build_efficientnet()
        )

        self.ycbcr_backbone, _ = (
            build_efficientnet()
        )

        self.rgb_projection = nn.Sequential(
    nn.Linear(feature_dim, token_dim),
    nn.LayerNorm(token_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)

        self.hsv_projection = nn.Sequential(
    nn.Linear(feature_dim, token_dim),
    nn.LayerNorm(token_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)

        self.ycbcr_projection = nn.Sequential(
    nn.Linear(feature_dim, token_dim),
    nn.LayerNorm(token_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)

        self.branch_gate = nn.Sequential(
            nn.Linear(
                token_dim * 3,
                token_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                0.25
            ),
            nn.Linear(
                token_dim,
                3,
            ),
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.cross_color_transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.fusion_score = nn.Sequential(
            nn.Linear(
                token_dim,
                token_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                token_dim // 2,
                1,
            ),
        )

        self.rgb_residual = nn.Sequential(
    nn.Linear(token_dim, token_dim),
    nn.LayerNorm(token_dim),
)

        self.residual_gate = nn.Sequential(
            nn.Linear(
                token_dim * 2,
                token_dim,
            ),
            nn.GELU(),
            nn.Linear(
                token_dim,
                token_dim,
            ),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(
            token_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                token_dim,
                256,
            ),
            nn.LayerNorm(
                256
            ),
            nn.GELU(),
            nn.Dropout(
                0.35
            ),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        rgb = x

        hsv = rgb_to_hsv(x)

        ycbcr = rgb_to_ycbcr(x)

        rgb = normalize(
            rgb,
            RGB_MEAN,
            RGB_STD,
        )

        hsv = normalize(
            hsv,
            HSV_MEAN,
            HSV_STD,
        )

        ycbcr = normalize(
            ycbcr,
            YCBCR_MEAN,
            YCBCR_STD,
        )

        rgb_features = (
            self.rgb_backbone(
                rgb
            )
        )

        hsv_features = (
            self.hsv_backbone(
                hsv
            )
        )

        ycbcr_features = (
            self.ycbcr_backbone(
                ycbcr
            )
        )

        rgb_token = (
            self.rgb_projection(
                rgb_features
            )
        )

        hsv_token = (
            self.hsv_projection(
                hsv_features
            )
        )

        ycbcr_token = (
            self.ycbcr_projection(
                ycbcr_features
            )
        )

        tokens = torch.stack(
            [
                rgb_token,
                hsv_token,
                ycbcr_token,
            ],
            dim=1,
        )

        combined = torch.cat(
            [
                rgb_token,
                hsv_token,
                ycbcr_token,
            ],
            dim=1,
        )

        branch_logits = (
            self.branch_gate(
                combined
            )
        )

        branch_weights = torch.softmax(
            branch_logits,
            dim=1,
        )

        gated_tokens = (
            tokens
            * branch_weights.unsqueeze(
                -1
            )
        )

        attended = (
            self.cross_color_transformer(
                gated_tokens
            )
        )

        fusion_scores = (
            self.fusion_score(
                attended
            )
        )

        fusion_weights = torch.softmax(
            fusion_scores,
            dim=1,
        )

        fused_color = (
            attended
            * fusion_weights
        ).sum(
            dim=1
        )

        rgb_residual = (
            self.rgb_residual(
                rgb_token
            )
        )

        fusion_input = torch.cat(
            [
                fused_color,
                rgb_residual,
            ],
            dim=1,
        )

        residual_gate = (
            self.residual_gate(
                fusion_input
            )
        )

        fused = (
            residual_gate
            * fused_color
            +
            (
                1.0
                - residual_gate
            )
            * rgb_residual
        )

        fused = self.fusion_norm(
            fused
        )

        logits = self.classifier(
            fused
        )

        return (
            logits,
            branch_weights,
        )


# ============================================================
# CSTF-NET
# ============================================================

class DepthwiseSeparableConv(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.GELU()

    def forward(
        self,
        x,
    ):

        x = self.depthwise(x)

        x = self.pointwise(x)

        x = self.bn(x)

        x = self.act(x)

        return x


class ColorBranch(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),

            DepthwiseSeparableConv(
                32,
                64,
                stride=2,
            ),

            DepthwiseSeparableConv(
                64,
                128,
                stride=2,
            ),

            DepthwiseSeparableConv(
                128,
                256,
                stride=2,
            ),

            DepthwiseSeparableConv(
                256,
                256,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d(
                1
            ),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                0.15
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.projection(
            self.features(x)
        )


class TextureBranch(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),

            DepthwiseSeparableConv(
                32,
                64,
                stride=2,
            ),

            DepthwiseSeparableConv(
                64,
                128,
                stride=2,
            ),

            DepthwiseSeparableConv(
                128,
                256,
                stride=2,
            ),

            DepthwiseSeparableConv(
                256,
                256,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d(
                1
            ),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                0.15
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.projection(
            self.features(x)
        )


class CSTF_Net(
    nn.Module
):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()

        rgb_model = models.efficientnet_b0(
            weights=None
        )

        rgb_features = (
            rgb_model
            .classifier[1]
            .in_features
        )

        rgb_model.classifier = (
            nn.Identity()
        )

        self.rgb_branch = (
            rgb_model
        )

        self.rgb_projection = nn.Sequential(
            nn.Linear(
                rgb_features,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
        )

        self.color_branch = (
            ColorBranch()
        )

        self.texture_branch = (
            TextureBranch()
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(
                128 * 3,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                128,
                3,
            ),
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(
                128 * 2,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                128,
            ),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                128
            ),
            nn.Linear(
                128,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                0.30
            ),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        rgb = self.rgb_branch(
            x
        )

        rgb = self.rgb_projection(
            rgb
        )

        hsv = rgb_to_hsv(
            x
        )

        color = self.color_branch(
            hsv
        )

        texture_map = sobel_texture(
            x
        )

        texture = self.texture_branch(
            texture_map
        )

        tokens = torch.stack(
            [
                rgb,
                color,
                texture,
            ],
            dim=1,
        )

        combined = torch.cat(
            [
                rgb,
                color,
                texture,
            ],
            dim=1,
        )

        gate_logits = (
            self.branch_gate(
                combined
            )
        )

        branch_weights = torch.softmax(
            gate_logits,
            dim=1,
        )

        weighted_tokens = (
            tokens
            * branch_weights.unsqueeze(
                -1
            )
        )

        transformed = (
            self.transformer(
                weighted_tokens
            )
        )

        transformed_global = (
            transformed.mean(
                dim=1
            )
        )

        original_global = (
            weighted_tokens.mean(
                dim=1
            )
        )

        fusion_input = torch.cat(
            [
                original_global,
                transformed_global,
            ],
            dim=1,
        )

        fusion_gate = (
            self.fusion_gate(
                fusion_input
            )
        )

        fused = (
            fusion_gate
            * transformed_global
            +
            (
                1.0
                - fusion_gate
            )
            * original_global
        )

        logits = self.classifier(
            fused
        )

        return (
            logits,
            branch_weights,
        )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    cleaned = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return checkpoint, cleaned


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(
    path,
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    return image


def image_to_tensor(
    image,
):

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    array = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1,
    )

    return tensor.unsqueeze(
        0
    )


def collect_images(
    root,
):

    files = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            files.append(
                path
            )

    return files


def select_random_images(
    root,
    count,
):

    files = collect_images(
        root
    )

    if len(files) < count:

        raise RuntimeError(
            f"Only {len(files)} images found "
            f"in {root}"
        )

    return random.sample(
        files,
        count,
    )


# ============================================================
# CREATE FUSED LAYERCAM
# ============================================================

def normalize_cam(
    cam,
):

    cam = np.maximum(
        cam,
        0
    )

    cam -= cam.min()

    maximum = cam.max()

    if maximum > 1e-8:
        cam /= maximum

    return cam


def fuse_branch_cams(
    cams,
    branch_weights,
):

    """
    Weighted fusion of branch LayerCAM maps.

    cams:
        list of [H,W]

    branch_weights:
        [3]

    """

    fused = np.zeros_like(
        cams[0],
        dtype=np.float32,
    )

    for cam, weight in zip(
        cams,
        branch_weights,
    ):

        fused += (
            float(weight)
            * normalize_cam(cam)
        )

    return normalize_cam(
        fused
    )


def make_overlay(
    image,
    cam,
    title,
):

    original = np.asarray(
        image.convert("RGB")
    )

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    cam_uint8 = np.uint8(
        np.clip(
            cam,
            0,
            1,
        )
        * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        0.55
        * original.astype(
            np.float32
        )
        +
        0.45
        * heatmap.astype(
            np.float32
        )
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Original panel
    # --------------------------------------------------------

    original_panel = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR,
    )

    overlay_panel = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR,
    )

    label_height = 60

    original_panel = cv2.copyMakeBorder(
        original_panel,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    overlay_panel = cv2.copyMakeBorder(
        overlay_panel,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    cv2.putText(
        original_panel,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay_panel,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.hconcat(
        [
            original_panel,
            overlay_panel,
        ]
    )


# ============================================================
# EYE FUSED LAYERCAM
# ============================================================

def generate_eye_layercam(
    image_path,
    model,
    classes,
):

    image = load_image(
        image_path
    )

    tensor = image_to_tensor(
        image
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # Three branch LayerCAM hooks
    # --------------------------------------------------------

    rgb_cam = LayerCAM(
        model,
        model.rgb_backbone.features[-1],
    )

    hsv_cam = LayerCAM(
        model,
        model.hsv_backbone.features[-1],
    )

    ycbcr_cam = LayerCAM(
        model,
        model.ycbcr_backbone.features[-1],
    )

    try:

        logits, branch_weights = (
            model(tensor)
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )

        # ----------------------------------------------------
        # LayerCAM for each color branch
        # ----------------------------------------------------

        rgb_map = rgb_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        hsv_map = hsv_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        ycbcr_map = ycbcr_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        weights = (
            branch_weights[0]
            .detach()
            .cpu()
            .numpy()
        )

        fused_cam = fuse_branch_cams(
            [
                rgb_map,
                hsv_map,
                ycbcr_map,
            ],
            weights,
        )

        class_name = classes[
            predicted_index
        ]

        return (
            image,
            fused_cam,
            class_name,
            confidence,
            weights,
        )

    finally:

        rgb_cam.remove()
        hsv_cam.remove()
        ycbcr_cam.remove()


# ============================================================
# SKIN FUSED LAYERCAM
# ============================================================

def generate_skin_layercam(
    image_path,
    model,
    classes,
):

    image = load_image(
        image_path
    )

    tensor = image_to_tensor(
        image
    ).to(
        DEVICE
    )

    rgb_cam = LayerCAM(
        model,
        model.rgb_branch.features[-1],
    )

    color_cam = LayerCAM(
        model,
        model.color_branch.features[-1],
    )

    texture_cam = LayerCAM(
        model,
        model.texture_branch.features[-1],
    )

    try:

        logits, branch_weights = (
            model(tensor)
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )

        rgb_map = rgb_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        color_map = color_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        texture_map = texture_cam.generate(
            logits,
            predicted_index,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        weights = (
            branch_weights[0]
            .detach()
            .cpu()
            .numpy()
        )

        fused_cam = fuse_branch_cams(
            [
                rgb_map,
                color_map,
                texture_map,
            ],
            weights,
        )

        class_name = classes[
            predicted_index
        ]

        return (
            image,
            fused_cam,
            class_name,
            confidence,
            weights,
        )

    finally:

        rgb_cam.remove()
        color_cam.remove()
        texture_cam.remove()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LAYERCAM QUALITATIVE ANALYSIS")
    print("=" * 70)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Eye test directory  : {EYE_TEST_DIR}"
    )

    print(
        f"Skin test directory : {SKIN_TEST_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load eye checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading MCFA-Net V2..."
    )

    eye_checkpoint, eye_state = (
        load_checkpoint(
            EYE_CHECKPOINT
        )
    )

    eye_classes = (
        eye_checkpoint.get(
            "class_names"
        )
    )

    if eye_classes is None:

        eye_classes = [
            "Blepharitis",
            "Conjunctivitis",
            "Corneal_Sequestrum",
            "Corneal_Ulcer",
            "Health",
            "Non_ulcerative",
            "Cataract",
            "Cherry_Eye",
            "Glaucoma",
        ]

    eye_model = MCFA_Net_V2(
        num_classes=len(
            eye_classes
        )
    )

    eye_model.load_state_dict(
        eye_state,
        strict=False,
    )

    eye_model = eye_model.to(
        DEVICE
    )

    eye_model.eval()

    # --------------------------------------------------------
    # Load skin checkpoint
    # --------------------------------------------------------

    print(
        "Loading CSTF-Net..."
    )

    skin_checkpoint, skin_state = (
        load_checkpoint(
            SKIN_CHECKPOINT
        )
    )

    skin_classes = (
        skin_checkpoint.get(
            "class_names"
        )
    )

    if skin_classes is None:

        skin_classes = [
            "Bacterial_dermatosis",
            "demodicosis",
            "Dermatitis",
            "Flea Allergy",
            "Fungal_infections",
            "hotspot",
            "Hypersensitivity",
            "Hypersensitivity_alergic_dermatosis",
            "mange",
            "ringworm",
            "Scabies",
        ]

    skin_model = CSTF_Net(
        num_classes=len(
            skin_classes
        )
    )

    skin_model.load_state_dict(
        skin_state,
        strict=False,
    )

    skin_model = skin_model.to(
        DEVICE
    )

    skin_model.eval()

    # --------------------------------------------------------
    # Random selection
    # --------------------------------------------------------

    eye_images = (
        select_random_images(
            EYE_TEST_DIR,
            NUM_RANDOM_IMAGES,
        )
    )

    skin_images = (
        select_random_images(
            SKIN_TEST_DIR,
            NUM_RANDOM_IMAGES,
        )
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "seed": SEED,
        "eye_images": [],
        "skin_images": [],
    }

    # --------------------------------------------------------
    # Eye examples
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING EYE FUSED LAYERCAM"
    )

    print(
        "=" * 70
    )

    for index, image_path in enumerate(
        eye_images,
        start=1,
    ):

        print(
            f"\nEye example {index}/5"
        )

        print(
            f"Image: {image_path.name}"
        )

        (
            image,
            cam,
            prediction,
            confidence,
            weights,
        ) = generate_eye_layercam(
            image_path,
            eye_model,
            eye_classes,
        )

        output_path = (
            OUTPUT_DIR
            / "eye"
            / f"{index:02d}_{image_path.stem}"
            / "mcfa_v2_layercam.png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        title = (
            f"MCFA-Net V2 LayerCAM | "
            f"{prediction} | "
            f"{confidence * 100:.1f}%"
        )

        result = make_overlay(
            image,
            cam,
            title,
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                result,
                cv2.COLOR_RGB2BGR,
            ),
        )

        metadata[
            "eye_images"
        ].append(
            {
                "image": str(
                    image_path
                ),
                "prediction": prediction,
                "confidence": confidence,
                "branch_weights": {
                    "RGB": float(
                        weights[0]
                    ),
                    "HSV": float(
                        weights[1]
                    ),
                    "YCbCr": float(
                        weights[2]
                    ),
                },
                "output": str(
                    output_path
                ),
            }
        )

        print(
            f"Prediction : {prediction}"
        )

        print(
            f"Confidence : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Output     : {output_path}"
        )

    # --------------------------------------------------------
    # Skin examples
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING SKIN FUSED LAYERCAM"
    )

    print(
        "=" * 70
    )

    for index, image_path in enumerate(
        skin_images,
        start=1,
    ):

        print(
            f"\nSkin example {index}/5"
        )

        print(
            f"Image: {image_path.name}"
        )

        (
            image,
            cam,
            prediction,
            confidence,
            weights,
        ) = generate_skin_layercam(
            image_path,
            skin_model,
            skin_classes,
        )

        output_path = (
            OUTPUT_DIR
            / "skin"
            / f"{index:02d}_{image_path.stem}"
            / "cstf_layercam.png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        title = (
            f"CSTF-Net LayerCAM | "
            f"{prediction} | "
            f"{confidence * 100:.1f}%"
        )

        result = make_overlay(
            image,
            cam,
            title,
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                result,
                cv2.COLOR_RGB2BGR,
            ),
        )

        metadata[
            "skin_images"
        ].append(
            {
                "image": str(
                    image_path
                ),
                "prediction": prediction,
                "confidence": confidence,
                "branch_weights": {
                    "RGB": float(
                        weights[0]
                    ),
                    "HSV": float(
                        weights[1]
                    ),
                    "Texture": float(
                        weights[2]
                    ),
                },
                "output": str(
                    output_path
                ),
            }
        )

        print(
            f"Prediction : {prediction}"
        )

        print(
            f"Confidence : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Output     : {output_path}"
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR /
        "selected_examples.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSelected:"
    )

    print(
        "  Eye  : 5 random test images"
    )

    print(
        "  Skin : 5 random test images"
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_path}"
    )


if __name__ == "__main__":
    main()