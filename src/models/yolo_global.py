import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.models.base_model import BaseModel
from src.models.constants import YOLO_GLOBAL_MODELS, DETECTION, SEGMENTATION
import json


def pred_yolo_global(detector, class_map, args):
    input_root = args.seg_gt if args.model.startswith("YOLO-SEG") else args.bbox_gt
    return detector.process_directory(
        input_root=input_root,
        model_name=args.model_type,
        class_map=class_map,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        output_name=args.output_name,
        pred_only=args.save_predictions_only,
        image_size=args.image_size,
        metrics_only=args.save_metrics_only,
    )


def add_yolo_global_parser(subparsers, parent_parser, train=False, optim=False):
    yolo_parser = subparsers.add_parser('yolo_global', help='Use the custom YOLO-Global model.', parents=[parent_parser])
    yolo_parser.add_argument('--model', type=str, default='YOLO-X', choices=YOLO_GLOBAL_MODELS.keys())
    
    if not optim:
        yolo_parser.add_argument('--score-threshold', type=float, default=0.05, help='Detection confidence threshold.')
    
    yolo_parser.set_defaults(load_func=YoloGlobalDetector.load_detector)

    if train:
        yolo_parser.add_argument('--yaml-path', type=str, help='Path to the Ultralytics dataset configuration YAML.')
    else:
        yolo_parser.set_defaults(func=pred_yolo_global)


class YoloGlobalDetector(BaseModel):
    """
    Detector class for the Custom YOLO-Global model.
    """
    def __init__(self, model_id):
        pred_mode = DETECTION if 'seg' not in model_id.lower() else SEGMENTATION
        self.model_type = pred_mode
        super().__init__(model_id)
    
    
    def load_model(self, model_id):
        """
        Loads the YOLOGlobal model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        model = YOLO(model_id)
        
        # Patch attention modules that may be missing the save_attention attribute
        # (compatibility fix for models trained with older ultralytics versions)
        for module in model.model.modules():
            if module.__class__.__name__ in ['GAM', 'SimAM']:
                if not hasattr(module, 'save_attention'):
                    module.save_attention = False
        
        return model, None


    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Custom YOLO-Global model.
        The model's classes should be set once before calling this method.

        The model inference time is extracted from Ultralytic's result object.
        For further information, see here:
        https://github.com/ultralytics/ultralytics/blob/8f5665717cbc1d88c4b24934dc91f399a891aead/ultralytics/engine/results.py#L262
        """
        score_threshold = kwargs.get('score_threshold')

        if not score_threshold:
            print("Argument score_threshold not specified. Using default value (0.05)")
            score_threshold = 0.05

        batch_results = self.model.predict(images, conf=score_threshold, verbose=False)

        batch_res_data = []
        batch_res_info = []

        for img_result in batch_results:
            # Get the mapping from class index to class name (prompt) for this result
            names = img_result.names

            img_res_data = []
            img_res_info = {
                "preprocess_time": img_result.speed["preprocess"],
                "inference_time": img_result.speed["inference"],
                "postprocess_time": img_result.speed["postprocess"]
            }

            if self.model_type == DETECTION:
                for box in img_result.boxes:
                    class_id_tensor = box.cls
                    # Ensure a class was detected for the bounding box
                    if class_id_tensor.numel() == 0:
                        continue

                    class_id = int(class_id_tensor[0])
                    label = names[class_id]
                    score = float(box.conf[0])
                    
                    # The .xyxy attribute provides box coordinates in [xmin, ymin, xmax, ymax] format
                    bounding_box = box.xyxy[0].tolist()

                    # Ensure the detected label is one of the prompts we care about
                    if label in class_map:
                        img_res_data.append({
                            "score": score,
                            "label": label,
                            "box": bounding_box,
                            "class_index": class_map[label]
                        })
            elif self.model_type == SEGMENTATION:
                masks = img_result.masks
                boxes = img_result.boxes
                
                if masks is None:
                    img_res_info["error"] = "No mask found in the image"
                    batch_res_data.append([])
                    batch_res_info.append(img_res_info)
                    continue
                
                h, w = masks.orig_shape

                for mask, box in zip(masks.data, boxes):
                    class_id_tensor = box.cls
                    # Ensure a class was detected for the mask
                    if class_id_tensor.numel() == 0:
                        continue

                    class_id = int(class_id_tensor[0])
                    label = names[class_id]
                    score = float(box.conf[0])
                    
                    # Convert the mask tensor to a binary mask
                    binary_mask = (mask.cpu().numpy() > 0.5).astype('uint8')

                    # Resize mask to original image dimensions if necessary
                    if binary_mask.shape != (h, w):
                        mask_img = Image.fromarray(binary_mask)
                        mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
                        binary_mask = np.array(mask_img).astype('uint8')
                    
                    # The .xyxy attribute provides box coordinates in [xmin, ymin, xmax, ymax] format
                    bounding_box = box.xyxy[0].tolist()
                    
                    # Ensure the detected label is one of the prompts we care about
                    if label in class_map:
                        img_res_data.append({
                            "score": score,
                            "label": label,
                            "box": bounding_box,
                            "mask": binary_mask,
                            "class_index": class_map[label]
                        })
            
            batch_res_data.append(img_res_data)
            batch_res_info.append(img_res_info)
            
        return batch_res_data, batch_res_info


    def train(self, args):
        """
        Trains the YOLO-World model using the Ultralytics library's built-in training method.
        Additional training parameters can be passed via kwargs.
        """
        
        metrics = self.model.train(
            data=args.yaml_path,
            batch=args.batch_size,
            imgsz=args.image_size,
        )

        if metrics:
            print(json.dumps(metrics, indent=4))
    

    @staticmethod
    def load_detector(args):
        print("Loading Custom YOLO-Global detector...")
        model_id = YOLO_GLOBAL_MODELS[args.model]
        class_map = {k: 0 for k in args.class_names}
        detector = YoloGlobalDetector(model_id)
        
        # print(f"Setting model classes to: {list(class_map.keys())}")
        # detector.model.set_classes(list(class_map.keys()))
        return detector, class_map

