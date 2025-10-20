from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM
from src.detectors.base_detector import BaseDetector
from src.metrics import SegmentationMetrics
import torch
import numpy as np

class GroundedSamDetector(BaseDetector):
    """
    Detector class for the Grounded SAM pipeline.
    Uses Transformers for Grounding DINO and Ultralytics for SAM.
    """
    def __init__(self, model_id):
        self.metrics = SegmentationMetrics
        super().__init__(model_id)
    

    def load_model(self, model_id):
        """
        Loads the Grounding DINO model from Transformers and SAM from Ultralytics.
        """
        print("Loading Grounding DINO model...")
        dino_processor = AutoProcessor.from_pretrained(model_id['dino'])
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id['dino']).to(self.device)
        
        print("Loading Ultralytics SAM model...")
        sam_model = SAM(model_id['sam'])
        sam_model.to(self.device)

        return {"dino": dino_model, "sam": sam_model}, dino_processor
    

    def predict(self, images, class_map, precomputed_boxes=None, **kwargs):
        """
        Performs the two-stage prediction: DINO for boxes, Ultralytics SAM for masks.
        If precomputed_boxes are provided, it skips the DINO step.
        """
        
        if precomputed_boxes:
            # MODE 1: Use pre-computed boxes from YOLO txt files
            # Create a placeholder structure that mimics DINO's output
            dino_results_raw = []
            for boxes in precomputed_boxes:
                dino_results_raw.append({
                    'boxes': torch.tensor(boxes, device=self.device) if boxes else torch.empty(0, 4, device=self.device),
                    'scores': torch.tensor([1.0] * len(boxes), device=self.device), # Placeholder score
                    'text_labels': [list(class_map.keys())[0]] * len(boxes) # Placeholder label
                })
        else:
            # MODE 2: Run Grounding DINO to get boxes
            text_for_model = list(class_map.keys())
            text_threshold = kwargs.get('text_threshold', 0.3)
            box_threshold = kwargs.get('threshold', 0.35)
            text_inputs = [text_for_model] * len(images)

            # --- 1. Grounding DINO Prediction (Batched) ---
            dino_inputs = self.processor(
                images=images, text=text_inputs, return_tensors="pt", padding=True
            ).to(self.device)
            
            with torch.no_grad():
                dino_outputs = self.model["dino"](**dino_inputs)

            target_sizes = [image.size[::-1] for image in images]
            dino_results_raw = self.processor.post_process_grounded_object_detection(
                dino_outputs, dino_inputs.input_ids, target_sizes=target_sizes,
                threshold=box_threshold, text_threshold=text_threshold
            )

        all_processed_results = []
        # --- 2. SAM Prediction (Iterative) ---
        for image, dino_results in zip(images, dino_results_raw):
            processed_for_image = []
            
            if not dino_results['boxes'].nelement():
                all_processed_results.append(processed_for_image)
                continue

            image_np = np.array(image)
            
            # The ultralytics SAM model expects boxes in a tensor.
            # The clone prevents the SAM model from modifying our original dino_results tensor.
            boxes_for_sam = dino_results["boxes"].clone().to(self.device)
            
            if boxes_for_sam.nelement():
                sam_results = self.model["sam"](image_np, bboxes=boxes_for_sam, verbose=False)
                
                if sam_results and sam_results[0].masks:
                    sam_masks = sam_results[0].masks.data.cpu().numpy()
                    num_detections = len(dino_results["scores"])
                    num_masks = len(sam_masks)

                    if num_detections != num_masks:
                        print(f"WARNING: Mismatch for image. DINO found {num_detections} boxes, but SAM returned {num_masks} masks. Processing common subset.")
                    
                    num_to_process = min(num_detections, num_masks)
                    for i in range(num_to_process):
                        score = dino_results["scores"][i]
                        label = dino_results["text_labels"][i]
                        box = dino_results["boxes"][i]
                        
                        if label in class_map:
                            processed_for_image.append({
                                "score": score.item(),
                                "label": label,
                                "box": box.tolist(),
                                "class_index": class_map[label],
                                "mask": sam_masks[i].squeeze().astype(np.uint8)
                            })
            
            all_processed_results.append(processed_for_image)
            
        return all_processed_results

