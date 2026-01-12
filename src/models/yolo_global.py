import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.models.base_model import BaseModel
from src.models.constants import YOLO_GLOBAL_MODELS, DETECTION, SEGMENTATION
import json


def pred_yolo_global(detector, class_map, args):
    return detector.process_directory(
        input_root=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        output_name=args.output_name,
        pred_only=args.save_predictions_only,
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
        return model, None


    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Custom YOLO-Global model.
        The model's classes should be set once before calling this method.
        """
        score_threshold = kwargs.get('score_threshold')

        if not score_threshold:
            print("Argument score_threshold not specified. Using default value (0.05)")
            score_threshold = 0.05

        results_batch = self.model.predict(images, conf=score_threshold, verbose=False)

        all_processed_results = []
        for result in results_batch:
            processed_for_image = []
            # Get the mapping from class index to class name (prompt) for this result
            names = result.names

            if self.model_type == DETECTION:
                for box in result.boxes:
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
                        processed_for_image.append({
                            "score": score,
                            "label": label,
                            "box": bounding_box,
                            "class_index": class_map[label] # This will always be 0
                        })
            elif self.model_type == SEGMENTATION:
                masks = result.masks
                
                if masks is None:
                    all_processed_results.append(processed_for_image)
                    continue
                
                h, w = masks.orig_shape

                for i, mask in enumerate(masks.data):
                    # TODO enable multiclass and scoring
                    # class_id_tensor = mask.cls
                    # # Ensure a class was detected for the mask
                    # if class_id_tensor.numel() == 0:
                    #     continue

                    class_id = 0 # int(class_id_tensor[0])
                    label = names[class_id]
                    score = 0.0 # float(result.masks.conf[i])
                    
                    # Convert the mask tensor to a binary mask
                    binary_mask = (mask.cpu().numpy() > 0.5).astype('uint8')

                    # Resize mask to original image dimensions if necessary
                    if binary_mask.shape != (h, w):
                        mask_img = Image.fromarray(binary_mask)
                        mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
                        binary_mask = np.array(mask_img).astype('uint8')
                    
                    # Ensure the detected label is one of the prompts we care about
                    if label in class_map:
                        processed_for_image.append({
                            "score": score,
                            "label": label,
                            "mask": binary_mask,
                            "class_index": class_map[label] # This will always be 0
                        })
            
            all_processed_results.append(processed_for_image)
            
        return all_processed_results


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

