#!/usr/bin/env python3
"""
Build a single composite figure from one input image:
Original | Grad-CAM | LIME | SHAP (gradient-based) | activation attention | prediction summary.

CLI:
    python image.py path/to/photo.jpg
    python image.py path/to/photo.jpg -o explanations.png --lime-samples 500

Streamlit:
    streamlit run image.py
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import platform
from typing import Optional, Tuple

import streamlit as st

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lime.lime_image import LimeImageExplainer
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

try:
    import shap
except ImportError:
    shap = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLASSES = [
    "Vitamin A",
    "Vitamin B",
    "Vitamin C",
    "Vitamin D",
    "Vitamin E",
    "Retina Blood Vessel",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


def imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def pil_to_display_array(pil_img: Image.Image, size: int = 224) -> np.ndarray:
    arr = np.asarray(pil_img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_mobilenet_vitamin(
    weights_path: str = "mobilenet_vitamin.pth",
    classes: Optional[list] = None,
) -> Tuple[nn.Module, list]:
    classes = classes or DEFAULT_CLASSES
    num_classes = len(classes)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No %s found; using ImageNet backbone with fresh classifier head.", weights_path)
        pretrained = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.features.load_state_dict(pretrained.features.state_dict())
        nn.init.xavier_uniform_(model.classifier[1].weight)
        nn.init.zeros_(model.classifier[1].bias)

    model.to(device)
    model.eval()
    return model, classes


def predict_probs(model: nn.Module, pil_img: Image.Image) -> Tuple[torch.Tensor, int]:
    t = imagenet_transform()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(probs).item())
    return probs, pred


def _last_conv_layer(model: nn.Module) -> nn.Conv2d:
    last = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last


def compute_grad_cam(model: nn.Module, pil_img: Image.Image, target_class: int) -> np.ndarray:
    """Return Grad-CAM map (H, W) in [0, 1]."""
    transform = imagenet_transform()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    layer = _last_conv_layer(model)
    activations: list = []
    gradients: list = []

    def fwd_hook(_m, _inp, out):
        activations.append(out)

    def bwd_hook(_m, _gi, go):
        gradients.append(go[0])
        return None

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        out = model(input_tensor)
        score = out[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture tensors.")

    act = activations[0]
    grad = gradients[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def compute_activation_attention(model: nn.Module, pil_img: Image.Image) -> np.ndarray:
    """Forward-only channel-mean |activation| map, upsampled to 224 (distinct from Grad-CAM)."""
    transform = imagenet_transform()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    layer = _last_conv_layer(model)
    captured: list = []

    def fwd_hook(_m, _inp, out):
        captured.append(out.detach())

    h = layer.register_forward_hook(fwd_hook)
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        h.remove()

    act = captured[0]
    att = act.abs().mean(dim=1, keepdim=True)
    att = torch.nn.functional.interpolate(att, size=(224, 224), mode="bilinear", align_corners=False)
    att = att.squeeze().cpu().numpy()
    att = (att - att.min()) / (att.max() - att.min() + 1e-8)
    return att


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """rgb and result in [0,1], shape (H,W,3)."""
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(h, colormap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(rgb * (1 - alpha) + color * alpha, 0, 1)


def compute_lime_mask(pil_img: Image.Image, model: nn.Module, classes: list, num_samples: int) -> np.ndarray:
    """Return LIME importance mask (H, W) float [0,1]."""
    img_np = np.array(pil_img.resize((224, 224), Image.Resampling.BILINEAR))

    def predict_fn(images):
        model.eval()
        tfm = imagenet_transform()
        batch = torch.stack([tfm(Image.fromarray(x.astype(np.uint8))) for x in images]).to(device)
        with torch.no_grad():
            logits = model(batch)
            return torch.softmax(logits, dim=1).cpu().numpy()

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=len(classes),
        hide_color=0,
        num_samples=num_samples,
    )
    _temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=True,
    )
    m = np.asarray(mask, dtype=np.float32)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m


def compute_shap_saliency(model: nn.Module, pil_img: Image.Image, target_class: int) -> Optional[np.ndarray]:
    """GradientExplainer-based saliency for the predicted class; returns (H,W) in [0,1]."""
    if shap is None:
        logger.warning("shap not installed; skipping SHAP panel.")
        return None

    use_device = device
    if use_device.type == "mps":
        use_device = torch.device("cpu")
        model_cpu = model.to(use_device)
    else:
        model_cpu = model

    try:
        tfm = imagenet_transform()
        x = tfm(pil_img).unsqueeze(0).to(use_device)
        background = torch.rand(8, 3, 224, 224, device=use_device) * 0.5

        explainer = shap.GradientExplainer(model_cpu, background)
        shap_values = explainer.shap_values(x)

        if isinstance(shap_values, list):
            vals = shap_values[target_class]
        else:
            vals = shap_values

        arr = vals[0]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        sal = np.mean(np.abs(arr), axis=0)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal
    except Exception as e:
        logger.warning("SHAP computation failed (%s); panel will show a placeholder.", e)
        return None
    finally:
        if device.type == "mps":
            model.to(device)


def make_explanation_figure(
    pil_img: Image.Image,
    model: nn.Module,
    classes: list,
    lime_samples: int = 800,
    layout_four: bool = False,
) -> Tuple[plt.Figure, torch.Tensor, int]:
    """
    Build the matplotlib figure. Caller should save and `plt.close(fig)`, or use `figure_to_png_bytes(fig)`.
    Returns (figure, probs, predicted_index).
    """
    rgb = pil_to_display_array(pil_img)
    probs, pred = predict_probs(model, pil_img)
    target = pred

    grad_cam = compute_grad_cam(model, pil_img, target)
    lime_mask = compute_lime_mask(pil_img, model, classes, lime_samples)
    shap_map = compute_shap_saliency(model, pil_img, target)
    act_att = compute_activation_attention(model, pil_img)

    grad_overlay = overlay_heatmap(rgb, grad_cam, alpha=0.55)
    lime_overlay = overlay_heatmap(rgb, lime_mask, alpha=0.55, colormap=cv2.COLORMAP_HOT)
    if shap_map is not None:
        shap_overlay = overlay_heatmap(rgb, shap_map, alpha=0.55)
    else:
        shap_overlay = None
    att_cmap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)
    att_overlay = overlay_heatmap(rgb, act_att, alpha=0.55, colormap=att_cmap)

    pred_name = classes[target] if target < len(classes) else str(target)
    conf = float(probs[target].item())

    if layout_four:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        panels = [
            (axes[0, 0], rgb, "Original"),
            (axes[0, 1], grad_overlay, "Grad-CAM"),
            (axes[1, 0], lime_overlay, "LIME"),
            (axes[1, 1], shap_overlay if shap_overlay is not None else rgb, "SHAP" if shap_overlay is not None else "SHAP (unavailable)"),
        ]
        for ax, img, title in panels:
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"Predicted: {pred_name} ({conf:.1%})", fontsize=14, fontweight="bold", y=1.02)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        ax_flat = axes.ravel()
        specs = [
            (rgb, "Original"),
            (grad_overlay, "Grad-CAM"),
            (lime_overlay, "LIME"),
            (shap_overlay if shap_overlay is not None else rgb, "SHAP" if shap_overlay is not None else "SHAP (unavailable)"),
            (att_overlay, "Attention map\n(mean |activation|)"),
        ]
        for i, (img, title) in enumerate(specs):
            ax_flat[i].imshow(np.clip(img, 0, 1))
            ax_flat[i].set_title(title, fontsize=11, fontweight="bold")
            ax_flat[i].axis("off")

        ax_pred = ax_flat[5]
        ax_pred.axis("off")
        lines = [f"Prediction: {pred_name}", f"Confidence: {conf:.2%}", "", "Top classes:"]
        topk = torch.topk(probs, k=min(4, len(classes)))
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            lines.append(f"  {classes[idx]}: {p:.2%}")
        ax_pred.text(0.05, 0.95, "\n".join(lines), transform=ax_pred.transAxes, fontsize=11, verticalalignment="top", family="monospace")
        ax_pred.set_title("Prediction summary", fontsize=11, fontweight="bold")

        fig.suptitle("Explainability overview", fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    return fig, probs, pred


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_explanation_frame(
    image_path: str,
    out_path: str,
    lime_samples: int = 800,
    layout_four: bool = False,
) -> str:
    """
    layout_four: if True, 2x2 grid with Original, Grad-CAM, LIME, SHAP only (no separate attention panel).
    """
    pil_img = load_image(image_path)
    model, classes = load_mobilenet_vitamin()
    fig, _probs, _pred = make_explanation_figure(pil_img, model, classes, lime_samples, layout_four)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved composite figure to %s", out_path)
    return out_path


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


def run_streamlit_app() -> None:
    st.set_page_config(page_title="NutriScan — Explanation frame", layout="wide")
    st.title("Explainability frame")
    st.caption("Original, Grad-CAM, LIME, SHAP, and activation attention in one figure.")

    @st.cache_resource(show_spinner="Loading model…")
    def get_model_and_classes():
        return load_mobilenet_vitamin()

    with st.sidebar:
        st.header("Options")
        lime_samples = st.slider("LIME samples", min_value=200, max_value=2000, value=800, step=100)
        layout_four = st.checkbox("2×2 layout only (no attention / summary cell)", value=False)
        st.divider()
        st.markdown("Run from terminal: `streamlit run image.py`")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if uploaded is None:
        st.info("Upload an image to generate the explanation frame.")
        return

    file_id = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    opts_key = (file_id, lime_samples, layout_four)

    pil_img = Image.open(uploaded).convert("RGB")
    model, classes = get_model_and_classes()

    if st.button("Generate explanation frame", type="primary"):
        with st.spinner("Computing Grad-CAM, LIME, SHAP, attention… (LIME can take a minute)"):
            try:
                fig, probs, pred = make_explanation_figure(
                    pil_img, model, classes, lime_samples=lime_samples, layout_four=layout_four
                )
                png_bytes = figure_to_png_bytes(fig)
            except Exception as e:
                st.error(f"Failed to build figure: {e}")
                logger.exception("Streamlit explanation frame failed")
                return

        st.session_state["xai_frame"] = {
            "opts_key": opts_key,
            "png": png_bytes,
            "pred": pred,
            "conf": float(probs[pred].item()),
        }

    cached = st.session_state.get("xai_frame")
    if cached and cached.get("opts_key") == opts_key:
        pred = cached["pred"]
        st.success(f"Top prediction: **{classes[pred]}** ({cached['conf']:.1%})")
        st.image(cached["png"], use_container_width=True)
        st.download_button(
            label="Download PNG",
            data=cached["png"],
            file_name="explanation_frame.png",
            mime="image/png",
            key="download_explanation_png",
        )
    elif uploaded is not None:
        st.caption("Click **Generate** to run explainability (results stay visible after reruns until settings change).")


def main():
    parser = argparse.ArgumentParser(description="Compose Original + Grad-CAM + LIME + SHAP + Attention into one figure.")
    parser.add_argument("image", help="Path to input image (jpg/png/...)")
    parser.add_argument("-o", "--output", default="explanation_frame.png", help="Output PNG path")
    parser.add_argument("--lime-samples", type=int, default=800, help="LIME perturbation count (higher = slower, more stable)")
    parser.add_argument(
        "--four-panel",
        action="store_true",
        help="2x2 layout: Original, Grad-CAM, LIME, SHAP only (no separate attention / summary cell).",
    )
    args = parser.parse_args()

    build_explanation_frame(
        args.image,
        args.output,
        lime_samples=args.lime_samples,
        layout_four=args.four_panel,
    )


if __name__ == "__main__":
    if _running_under_streamlit():
        run_streamlit_app()
    else:
        main()
