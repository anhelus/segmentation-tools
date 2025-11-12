from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM
from src.models.base_model import BaseModel
from src.models.constants import DINO_MODELS, SAM_MODELS, SEGMENTATION
import torch
import numpy as np


def pred_grounded_sam(detector, class_map, args):
    if args.precomputed_boxes:
        print(f"Running SAM only, using existing labels from: {args.precomputed_boxes}")
        
        # TODO da rivedere completamente
        return detector.process_precomputed_boxes(
            directory_path=args.bbox_gt,
            model_name=args.model_type,
            class_map=class_map,
            batch_size=args.batch_size
        )
    
    print("Running full Grounded SAM pipeline (DINO + SAM)...")

    return detector.process_directory(
        directory_path=args.seg_gt,
        model_name=args.model_type,
        class_map=class_map,
        batch_size=args.batch_size,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
        image_size=args.image_size,
        map_thresh_list=args.map_thresh
    )


def add_grounded_sam_parser(subparsers, parent_parser, train=False, optim=False):
    grounded_sam_parser = subparsers.add_parser('grounded_sam', help='Use Grounded SAM pipeline.', parents=[parent_parser])
    grounded_sam_parser.add_argument('--dino-model', type=str, default='GDINO-BASE', choices=DINO_MODELS.keys())
    grounded_sam_parser.add_argument('--sam-model', type=str, default='SAM-2.1', choices=SAM_MODELS.keys())
    grounded_sam_parser.add_argument(
        '--precomputed-boxes', action="store_true", default=None,
        help='Wether to skip the DINO step and use the boxes in dataset_path/bbox_ground_truth for SAM.'
    )

    if not optim:
        grounded_sam_parser.add_argument('--text-threshold', type=float, default=0.25, help='Confidence threshold for text matching.')
        grounded_sam_parser.add_argument('--box-threshold', type=float, default=0.2, help='Confidence threshold for object detection box.')
    
    grounded_sam_parser.set_defaults(load_func=GroundedSamDetector.load_detector)

    if not train:
        grounded_sam_parser.set_defaults(func=pred_grounded_sam)


class GroundedSamDetector(BaseModel):
    """
    Detector class for the Grounded SAM pipeline.
    Uses Transformers for Grounding DINO and Ultralytics for SAM.
    """
    def __init__(self, model_id):
        self.model_type = SEGMENTATION
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
    

    def model_identifier(self):
        dino_id = self.model_id["dino"].split('-')[-1]
        sam_id = self.model_id["sam"].split('/')[-1].replace(".pt", "")
        return f"grounded_sam[{dino_id}+{sam_id}]"
    

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
            text_threshold = kwargs.get('text_threshold', 0.25)
            box_threshold = kwargs.get('box_threshold', 0.2)
            text_inputs = [text_for_model] * len(images)

            # --- 1. Grounding DINO Prediction (Batched) ---
            dino_inputs = self.processor(
                images=images, text=text_inputs, return_tensors="pt", padding=True
            ).to(self.device)
            
            with torch.no_grad():
                dino_outputs = self.model["dino"](**dino_inputs)

            target_sizes = [image.size[::-1] for image in images]
            dino_results_raw = self.processor.post_process_grounded_object_detection(
                dino_outputs,
                dino_inputs.input_ids,
                target_sizes=target_sizes,
                threshold=box_threshold,
                text_threshold=text_threshold
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

                        mask = sam_masks[i].squeeze().astype(np.uint8)
                        
                        if label in class_map:
                            processed_for_image.append({
                                "score": score.item(),
                                "label": label,
                                "box": box.tolist(),
                                "class_index": class_map[label],
                                "mask": mask
                            })
            
            all_processed_results.append(processed_for_image)
            
        return all_processed_results


    def train(self, **kwargs):
        pass


    @staticmethod
    def load_detector(args):
        print("Initializing Grounded SAM detector...")
        model_id = {
            'dino': DINO_MODELS[args.dino_model],
            'sam': SAM_MODELS[args.sam_model]
        }
        class_map = {label: idx for idx, label in enumerate(args.class_names)}
        detector = GroundedSamDetector(model_id)

        return detector, class_map