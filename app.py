import gradio as gr
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
DINO_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weights/sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Initializing on {DEVICE}...")

# --- GLOBAL MODEL LOADING ---
# Models are loaded globally at startup to ensure low latency during inference.
# This keeps models in VRAM, requiring ~4GB total.
dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=DEVICE)

sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

print("odels loaded into VRAM.")

def transform_image(image_pil):
    """Transforms a PIL image into the specific Tensor format required by GroundingDINO"""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image

def run_pipeline(input_image, text_prompt, box_threshold, text_threshold):
    """
    Main pipeline combining detection and segmentation
    input_image: NumPy array from Gradio
    """
    # Preprocessing
    # Gradio provides NumPy, DINO requires PIL -> Tensor
    source_image = Image.fromarray(input_image).convert("RGB")
    dino_image = transform_image(source_image)

    # Grounding DINO Inference
    with torch.inference_mode():
        boxes, logits, phrases = predict(
            model=dino_model,
            image=dino_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

    # Early exit if no object is detected
    if boxes.numel() == 0:
        return input_image, " No objects detected."

    #Coordinate Transformation
    H, W, _ = input_image.shape
    boxes = boxes.to(DEVICE)
    scale_tensor = torch.tensor([W, H, W, H], device=DEVICE)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * scale_tensor

    # SAM Inference
    sam_predictor.set_image(input_image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, input_image.shape[:2]
    )

    # Use FP16 (Autocast) to leverage Tensor Cores on RTX 30xx
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

    # Annotation / Visualization
    boxes_np = boxes_xyxy.cpu().numpy()
    masks_np = masks.squeeze(1).cpu().numpy()

    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_np,
        class_id=np.array([0] * len(boxes))
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated_image = mask_annotator.annotate(scene=input_image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    return annotated_image, f"{len(boxes)} objects found."

# --- GRADIO INTERFACE ---
with gr.Blocks(title="Zero-Shot Segmentation Demo") as demo:
    gr.Markdown("#  Zero-Shot Segmentation (Grounded-SAM)")
    gr.Markdown("Detect and segment objects using open-vocabulary text prompts.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image")
            text_input = gr.Textbox(label="Text Prompt", value="dog")
            with gr.Accordion("Advanced Parameters", open=False):
                box_thresh = gr.Slider(0.0, 1.0, value=0.35, label="Box Threshold")
                text_thresh = gr.Slider(0.0, 1.0, value=0.25, label="Text Threshold")
            run_btn = gr.Button("Run Segmentation", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Result")
            status_text = gr.Label(label="Status")

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_img, text_input, box_thresh, text_thresh],
        outputs=[output_img, status_text]
    )

if __name__ == "__main__":
    # creating a public link valid for 72h
    demo.launch(share=True)