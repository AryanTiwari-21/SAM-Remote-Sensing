# ============================================================
# SAM Remote Sensing - Gradio UI (Enhanced)
# Mirrors Runner.ipynb exactly — same patches, paths, logic
#
# HOW TO USE:
# 1. Save this file to: MyDrive/DLminorproject/GradioApp.py
# 2. Open a NEW Colab notebook and run:
#
#    Cell 1: !pip install gradio -q
#    Cell 2: from google.colab import drive; drive.mount('/content/drive')
#    Cell 3: %run /content/drive/MyDrive/DLminorproject/GradioApp.py
# ============================================================

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import gradio as gr

# ── PATHS (exact same as Runner.ipynb) ──────────────────────
PROJECT_ROOT    = "/content/drive/MyDrive/DLminorproject"
SAM_CHECKPOINT  = f"{PROJECT_ROOT}/checkpoints/sam_vit_h_4b8939.pth"
GDINO_CONFIG    = f"{PROJECT_ROOT}/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS   = f"{PROJECT_ROOT}/checkpoints/groundingdino_swint_ogc.pth"
SEG_GEO         = f"{PROJECT_ROOT}/segment-geospatial"
GROUNDED_SAM    = f"{PROJECT_ROOT}/Grounded-Segment-Anything"

# ── SYS PATH ────────────────────────────────────────────────
sys.path.append(GROUNDED_SAM)
sys.path.append(f"{GROUNDED_SAM}/GroundingDINO")
sys.path.append(SEG_GEO)

# ── PATCH: transformers compatibility (Runner Cell 66) ──────
import transformers
from transformers.modeling_utils import PreTrainedModel

if not hasattr(transformers.models.bert.modeling_bert.BertModel, "get_head_mask"):
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        return [None] * num_hidden_layers
    transformers.models.bert.modeling_bert.BertModel.get_head_mask = get_head_mask

original_get_extended_attention_mask = PreTrainedModel.get_extended_attention_mask
def fixed_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
    if isinstance(dtype, torch.device):
        dtype = torch.float32
    return original_get_extended_attention_mask(self, attention_mask, input_shape, dtype)
PreTrainedModel.get_extended_attention_mask = fixed_get_extended_attention_mask
print("All patches applied ✅")

# ── LOAD SAM (Runner Cell 78) ────────────────────────────────
print("Loading SAM model...")
from segment_anything import sam_model_registry, SamPredictor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam_model.to(device)
predictor = SamPredictor(sam_model)
print("SAM loaded ✅")

# ── LOAD GROUNDINGDINO (Runner Cell 70, 72) ──────────────────
print("Loading GroundingDINO model...")
from groundingdino.util.inference import load_model, load_image, predict
gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
print("GroundingDINO loaded ✅")

# ── LOAD SAMGEO (Runner Cell 30) ─────────────────────────────
print("Loading SamGeo...")
from samgeo import SamGeo
sam_geo = SamGeo(model_type="vit_h", checkpoint=SAM_CHECKPOINT, automatic=True)
print("SamGeo loaded ✅")

print("\n🚀 All models ready!\n")

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def overlay_mask(image_rgb, mask, color=(0, 200, 100), alpha=0.45):
    result = image_rgb.copy()
    result[mask > 0] = (
        result[mask > 0] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    return result

def load_gt_mask(gt_mask_file):
    """Load and binarize GT mask — handles both LoveDA format and binary PNG."""
    gt_raw = np.array(Image.open(gt_mask_file).convert("L"))
    if gt_raw.max() <= 1:
        return (gt_raw == 1).astype(np.uint8)  # LoveDA: building = 1
    return (gt_raw > 0).astype(np.uint8)        # binary 0/255

def compute_iou(pred, gt):
    """Runner Cell 43/58/85/102/126."""
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 0

def evaluate_iou(pred_mask_bin, gt_mask_bin):
    """Auto-corrects mask inversion — same as Runner evaluate_iou."""
    if pred_mask_bin.shape != gt_mask_bin.shape:
        pred_mask_bin = cv2.resize(pred_mask_bin,
            (gt_mask_bin.shape[1], gt_mask_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    iou_normal   = compute_iou(pred_mask_bin, gt_mask_bin)
    iou_inverted = compute_iou(1 - pred_mask_bin, gt_mask_bin)
    return max(iou_normal, iou_inverted)

def iou_status(iou):
    """Human-readable IoU quality label."""
    if iou >= 0.7: return f"IoU: {iou:.3f} 🟢 Good"
    if iou >= 0.5: return f"IoU: {iou:.3f} 🟡 Moderate"
    return f"IoU: {iou:.3f} 🔴 Low"

# ════════════════════════════════════════════════════════════
# MODE 1 — Auto Segmentation (Runner Cell 32, 38, 39, 43)
# ════════════════════════════════════════════════════════════
def run_auto(image, gt_mask_file):
    try:
        tmp_in  = "/tmp/auto_input.png"
        tmp_out = "/tmp/auto_mask.png"
        Image.fromarray(image).save(tmp_in)

        sam_geo.generate(tmp_in, output=tmp_out)

        import rasterio
        with rasterio.open(tmp_out) as src:
            mask = src.read(1)

        result = overlay_mask(image, mask, color=(0, 200, 100))

        # IoU if GT provided
        iou_str = "No GT mask uploaded — IoU not computed"
        if gt_mask_file is not None:
            gt_bin   = load_gt_mask(gt_mask_file)
            pred_bin = (mask > 0).astype(np.uint8)
            iou      = evaluate_iou(pred_bin, gt_bin)
            iou_str  = iou_status(iou)

        return Image.fromarray(result), f"✅ Auto segmentation complete! | {iou_str}"
    except Exception as e:
        return Image.fromarray(image), f"❌ Error: {str(e)}"

# ════════════════════════════════════════════════════════════
# MODE 2 — Multi-Point Segmentation (Runner Cell 55, 58)
# ════════════════════════════════════════════════════════════
def run_multipoint(image, gt_mask_file):
    try:
        if gt_mask_file is None:
            return Image.fromarray(image), "⚠️ Please upload a ground truth mask."

        gt_bin = load_gt_mask(gt_mask_file)

        # Extract centroids — Runner Cell 55
        num_labels, labels = cv2.connectedComponents(gt_bin)
        points = []
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if len(xs) == 0:
                continue
            points.append([int(np.mean(xs)), int(np.mean(ys))])

        if len(points) == 0:
            return Image.fromarray(image), "⚠️ No buildings found in mask."

        input_point = np.array(points)
        input_label = np.ones(len(input_point))
        print(f"Total centroid points: {len(points)}")

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        point_mask = masks[0]

        result = overlay_mask(image, point_mask.astype(np.uint8), color=(0, 100, 255))
        for pt in points:
            cv2.circle(result, tuple(pt), 5, (0, 255, 0), -1)

        pred_bin = point_mask.astype(np.uint8)
        iou      = evaluate_iou(pred_bin, gt_bin)
        return Image.fromarray(result), f"✅ Multi-point done! | {iou_status(iou)} | Points used: {len(points)}"

    except Exception as e:
        return Image.fromarray(image), f"❌ Error: {str(e)}"

# ════════════════════════════════════════════════════════════
# MODE 3 — Text-Prompt Segmentation (Runner Cell 75, 79, 85)
# ════════════════════════════════════════════════════════════
def run_text(image, text_prompt, box_thresh, text_thresh, gt_mask_file):
    try:
        tmp_in = "/tmp/text_input.png"
        Image.fromarray(image).save(tmp_in)

        image_source, image_tensor = load_image(tmp_in)

        boxes, logits, phrases = predict(
            model=gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=float(box_thresh),
            text_threshold=float(text_thresh),
            device="cpu"
        )

        if len(boxes) == 0:
            return Image.fromarray(image), f"⚠️ No '{text_prompt}' detected. Try lower thresholds."

        H, W, _ = image_source.shape
        predictor.set_image(image_source)

        final_mask = np.zeros((H, W), dtype=np.uint8)
        for box, logit in zip(boxes, logits):
            if logit < 0.4:
                continue
            cx, cy, bw, bh = box.cpu().numpy()
            x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
            x2 = (cx + bw/2) * W; y2 = (cy + bh/2) * H
            box_xyxy = np.array([x1, y1, x2, y2])
            transformed_box = predictor.transform.apply_boxes(np.array([box_xyxy]), (H, W))
            mask, _, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=transformed_box[0], multimask_output=False)
            final_mask = np.logical_or(final_mask, mask[0]).astype(np.uint8)

        result = overlay_mask(image_source, final_mask, color=(255, 100, 0))
        for box in boxes:
            cx, cy, bw, bh = box.cpu().numpy()
            x1 = int((cx - bw/2) * W); y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W); y2 = int((cy + bh/2) * H)
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # IoU if GT provided
        iou_str = "No GT mask uploaded — IoU not computed"
        if gt_mask_file is not None:
            gt_bin  = load_gt_mask(gt_mask_file)
            iou     = evaluate_iou(final_mask, gt_bin)
            iou_str = iou_status(iou)

        return Image.fromarray(result), f"✅ Text-prompt done! | {len(boxes)} '{text_prompt}' detected | {iou_str}"

    except Exception as e:
        return Image.fromarray(image), f"❌ Error: {str(e)}"

# ════════════════════════════════════════════════════════════
# MODE 4 — Phase 4 Custom Modification (Runner Cell 111-126)
# ════════════════════════════════════════════════════════════
def run_phase4(image, box_thresh, text_thresh, gt_mask_file):
    try:
        tmp_in = "/tmp/phase4_input.png"
        Image.fromarray(image).save(tmp_in)

        image_source, image_tensor = load_image(tmp_in)
        H, W, _ = image_source.shape

        # Step 1: GroundingDINO — Runner Cell 111
        boxes, logits, _ = predict(
            model=gdino_model,
            image=image_tensor,
            caption="building",
            box_threshold=float(box_thresh),
            text_threshold=float(text_thresh),
            device="cpu"
        )

        if len(boxes) == 0:
            return Image.fromarray(image), "⚠️ No buildings detected. Try lowering thresholds."

        # Step 2: Filter small boxes — Runner Cell 113
        boxes_xyxy = []
        for box in boxes:
            cx, cy, bw, bh = box.cpu().numpy()
            x1 = int((cx - bw/2) * W); y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W); y2 = int((cy + bh/2) * H)
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                boxes_xyxy.append([x1, y1, x2, y2])

        # Step 3: Positive + negative points — Runner Cell 114
        points, labels_list = [], []
        for x1, y1, x2, y2 in boxes_xyxy:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bw, bh = x2-x1, y2-y1
            points.append([cx, cy]);                                          labels_list.append(1)
            points.append([max(0, x1-bw//2), max(0, y1-bh//2)]);             labels_list.append(0)
            points.append([min(W-1, x2+bw//2), min(H-1, y2+bh//2)]);        labels_list.append(0)

        input_points = np.array(points)
        input_labels = np.array(labels_list)

        # Step 4: SAM predict — Runner Cell 116
        predictor.set_image(image_source)
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask_uint8 = (masks[0] * 255).astype(np.uint8)

        # Step 5: Morphological refinement — Runner Cell 120
        kernel     = np.ones((7, 7), np.uint8)
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN,  kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Step 6: Remove small components — Runner Cell 121
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        final_mask = np.zeros_like(mask_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 500:
                final_mask[labels_map == i] = 255

        result = overlay_mask(image_source, final_mask, color=(200, 50, 255))

        # Draw boxes and points on result
        for x1, y1, x2, y2 in boxes_xyxy:
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
        for (x, y), lbl in zip(input_points, input_labels):
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
            cv2.circle(result, (int(x), int(y)), 5, color, -1)

        # IoU if GT provided
        iou_str = "No GT mask uploaded — IoU not computed"
        if gt_mask_file is not None:
            gt_bin  = load_gt_mask(gt_mask_file)
            pred_bin = (final_mask > 0).astype(np.uint8)
            iou     = evaluate_iou(pred_bin, gt_bin)
            iou_str = iou_status(iou)

        pos_count = int(np.sum(input_labels == 1))
        return Image.fromarray(result), \
               f"✅ Phase 4 done! | Boxes: {len(boxes_xyxy)} | +ve points: {pos_count} | {iou_str}"

    except Exception as e:
        return Image.fromarray(image), f"❌ Error: {str(e)}"

# ════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════
with gr.Blocks(theme=gr.themes.Soft(), title="SAM Remote Sensing") as demo:

    gr.Markdown("""
    # 🛰️ SAM for Remote Sensing — Interactive Demo
    **Building Segmentation on LoveDA Satellite Imagery**

    Upload a satellite image and choose a segmentation mode.
    Optionally upload a **Ground Truth mask** to compute **IoU score**.

    > 🟢 IoU ≥ 0.7 Good &nbsp;|&nbsp; 🟡 0.5–0.7 Moderate &nbsp;|&nbsp; 🔴 < 0.5 Low
    """)

    with gr.Tabs():

        # ── Tab 1: Auto Segmentation ──────────────────────────
        with gr.TabItem("🔍 Auto Segmentation"):
            gr.Markdown("""
            **Zero-shot**: no prompts needed. SAM segments all objects automatically.
            Optionally upload a GT mask to get IoU score.
            """)
            with gr.Row():
                a_img  = gr.Image(label="Upload Satellite Image", type="numpy")
                a_out  = gr.Image(label="Segmentation Result")
            a_gt       = gr.File(label="Upload Ground Truth Mask (.png) — Optional")
            a_status   = gr.Textbox(label="Status", interactive=False)
            gr.Button("▶ Run Auto Segmentation", variant="primary").click(
                run_auto, [a_img, a_gt], [a_out, a_status])

        # ── Tab 2: Multi-Point ────────────────────────────────
        with gr.TabItem("📍 Multi-Point Segmentation"):
            gr.Markdown("""
            **Required**: Upload both the satellite image and its ground truth mask.
            Building centroids are auto-extracted from the GT mask as SAM point prompts.
            IoU is computed automatically.
            """)
            with gr.Row():
                mp_img    = gr.Image(label="Upload Satellite Image", type="numpy")
                mp_out    = gr.Image(label="Segmentation Result")
            mp_mask   = gr.File(label="Upload Ground Truth Mask (.png) — Required")
            mp_status = gr.Textbox(label="Status", interactive=False)
            gr.Button("▶ Run Multi-Point Segmentation", variant="primary").click(
                run_multipoint, [mp_img, mp_mask], [mp_out, mp_status])

        # ── Tab 3: Text Prompt ────────────────────────────────
        with gr.TabItem("📝 Text-Prompt Segmentation"):
            gr.Markdown("""
            Type what to segment (e.g. `building`, `tree`, `road`).
            Adjust thresholds if nothing is detected.
            Optionally upload a GT mask to get IoU score.
            """)
            with gr.Row():
                t_img    = gr.Image(label="Upload Satellite Image", type="numpy")
                t_out    = gr.Image(label="Segmentation Result")
            with gr.Row():
                t_prompt = gr.Textbox(label="Text Prompt", value="building")
                t_box    = gr.Slider(0.1, 0.9, value=0.3,  step=0.05, label="Box Threshold")
                t_text   = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Text Threshold")
            t_gt         = gr.File(label="Upload Ground Truth Mask (.png) — Optional")
            t_status     = gr.Textbox(label="Status", interactive=False)
            gr.Button("▶ Run Text-Prompt Segmentation", variant="primary").click(
                run_text, [t_img, t_prompt, t_box, t_text, t_gt], [t_out, t_status])

        # ── Tab 4: Phase 4 ────────────────────────────────────
        with gr.TabItem("⚡ Phase 4 — Custom Modification"):
            gr.Markdown("""
            **Custom pipeline**: GroundingDINO detects boxes →
            box centers become positive SAM prompts, outside corners become negative →
            SAM segments → Morphological refinement removes noise.
            Optionally upload a GT mask to get IoU score.
            """)
            with gr.Row():
                p4_img   = gr.Image(label="Upload Satellite Image", type="numpy")
                p4_out   = gr.Image(label="Segmentation Result")
            with gr.Row():
                p4_box   = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Box Threshold")
                p4_text  = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Text Threshold")
            p4_gt        = gr.File(label="Upload Ground Truth Mask (.png) — Optional")
            p4_status    = gr.Textbox(label="Status", interactive=False)
            gr.Button("▶ Run Phase 4", variant="primary").click(
                run_phase4, [p4_img, p4_box, p4_text, p4_gt], [p4_out, p4_status])

    gr.Markdown("""
    ---
    **Project:** DL Minor Project &nbsp;|&nbsp;
    **Dataset:** LoveDA &nbsp;|&nbsp;
    **Models:** SAM ViT-H · GroundingDINO · PerSAM &nbsp;|&nbsp;
    **Authors:** Aryan Tiwari · Anant Pandey · Harsh Chhapre
    """)

demo.launch(share=True)
