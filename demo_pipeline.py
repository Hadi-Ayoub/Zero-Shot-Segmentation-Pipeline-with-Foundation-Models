import os
import torch
import cv2
import numpy as np
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from segment_anything import sam_model_registry, SamPredictor
import torch 


IMAGE_PATH = "/home/ayoub/DL/GroundingDINO/test_images/meeting.jpg"       
TEXT_PROMPT = "microphone"          
BOX_THRESHOLD = 0.35         
TEXT_THRESHOLD = 0.25

DINO_CONFIG = "/home/ayoub/DL/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "/home/ayoub/DL/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "/home/ayoub/DL/GroundingDINO/weights/sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"

DEVICE = torch.device("cuda")

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"{IMAGE_PATH} is not found ! ")
    image_source, image = load_image(IMAGE_PATH)
    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=DEVICE)

    print(f"Searching for : '{TEXT_PROMPT}'...")
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    print(f"{len(boxes)} objects detected")
    
    # VRAM optimisation 
    del dino_model
    torch.cuda.empty_cache

    print("Loading SAM ...")
    sam= sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    image_cv = cv2.imread(IMAGE_PATH)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)

    H, W, _ = image_cv.shape
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy.to(DEVICE), image_cv.shape[:2]
    )


    print("Segmenting ...")
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # visualizing
    boxes_np = boxes_xyxy.cpu().numpy()
    masks_np = masks.squeeze(1).cpu().numpy()

    # annotating
    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_np,
        class_id=np.array([0] * len(boxes)) # Tout est classe 0 pour l'instant
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated_image = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    # saving image 
    cv2.imwrite("rfinal_result.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()


