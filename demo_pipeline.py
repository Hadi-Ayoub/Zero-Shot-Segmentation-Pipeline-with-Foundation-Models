import os
import torch
import cv2
import numpy as np
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from segment_anything import sam_model_registry, SamPredictor
import torch 


IMAGE_SRC = "test_images"
TEXT_PROMPT = "dog"          
BOX_THRESHOLD = 0.35         
TEXT_THRESHOLD = 0.25

DINO_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weights/sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"

DEVICE = torch.device("cuda")

def main():

    results_file = "segmentation_output"
    os.makedirs(results_file, exist_ok = True)
    for f in os.listdir(results_file):
        os.remove(os.path.join(results_file, f))

    valid_ext = ('.jpg', '.jpeg', '.png')
    list_img_src = [f for f in os.listdir(IMAGE_SRC) if f.lower().endswith(valid_ext)]

    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=DEVICE)
    print(f"Searching for '{TEXT_PROMPT}' in {len(list_img_src)} images...")
    dino_results = {}

    with torch.inference_mode():
        for img_name in list_img_src:
            image_source, image = load_image(os.path.join(IMAGE_SRC, img_name))
            
            # GPU optimization : Autocast for FP16
            with torch.cuda.amp.autocast():
                boxes, logits, phrases = predict(
                    model=dino_model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )
            

            if boxes.numel() > 0:
                dino_results[img_name] = boxes
            else:
                print(f"Skipped {img_name} (No detection)")

    # VRAM cleaning
    del dino_model
    torch.cuda.empty_cache()



    ##### SAM #####

    print("Loading SAM ...")
    sam= sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)


    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    print("Segmentation loop...")
    

    with torch.inference_mode():
        for img_name, boxes in dino_results.items():
            
            full_path = os.path.join(IMAGE_SRC, img_name)
            image_cv = cv2.imread(full_path)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

            
            sam_predictor.set_image(image_cv)
            H, W, _ = image_cv.shape

            boxes = boxes.to(DEVICE)

            scale_tensor = torch.tensor([W, H, W, H], device=DEVICE)
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * scale_tensor
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_cv.shape[:2]
            )

            with torch.cuda.amp.autocast():
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

            # Visualization
            boxes_np = boxes_xyxy.cpu().numpy()
            masks_np = masks.squeeze(1).cpu().numpy()

            detections = sv.Detections(
                xyxy=boxes_np,
                mask=masks_np,
                class_id=np.array([0] * len(boxes))
            )

            # Annotation
            annotated_image = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            
            # Save
            output_path = os.path.join(results_file, f"res_{img_name}")
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"Processed {img_name}")


if __name__ == "__main__":
    main()


