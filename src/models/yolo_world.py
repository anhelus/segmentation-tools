from ultralytics import YOLOWorld
from src.models.base_model import BaseModel
from src.models.constants import YOLO_WORLD_MODELS, DETECTION
import json


def pred_yolo_world(detector, class_map, args):
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


def add_yolo_world_parser(subparsers, parent_parser, train=False, optim=False):
    yolo_parser = subparsers.add_parser('yolo_world', help='Use Ultralytics YOLO-World model.', parents=[parent_parser])
    yolo_parser.add_argument('--model', type=str, default='YOLO-X-WORLD', choices=YOLO_WORLD_MODELS.keys())
    yolo_parser.add_argument('--prompts', nargs='+', required=True, help='Descriptive text prompts for detection.')
    
    if not optim:
        yolo_parser.add_argument('--score-threshold', type=float, default=0.05, help='Detection confidence threshold.')
    
    yolo_parser.set_defaults(load_func=YoloWorldDetector.load_detector)

    if train:
        yolo_parser.add_argument('--yaml-path', type=str, help='Path to the Ultralytics dataset configuration YAML.')
    else:
        yolo_parser.set_defaults(func=pred_yolo_world)


class YoloWorldDetector(BaseModel):
    """
    Detector class for the YOLO-World model from the Ultralytics library.
    """
    def __init__(self, model_id):
        self.model_type = DETECTION
        super().__init__(model_id)
    
    
    def load_model(self, model_id):
        """
        Loads the YOLOWorld model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        model = YOLOWorld(model_id)
        return model, None


    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Ultralytics YOLO-World model.
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
        print("Loading Ultralytics YOLO-World detector...")
        model_id = YOLO_WORLD_MODELS[args.model]
        class_map = {k: 0 for k in args.prompts}
        detector = YoloWorldDetector(model_id)
        
        print(f"Setting model classes to: {list(class_map.keys())}")
        detector.model.set_classes(list(class_map.keys()))
        return detector, class_map

