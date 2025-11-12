from ultralytics import YOLOWorld
from src.models.base_model import BaseModel
from src.models.constants import YOLO_WORLD_MODELS, DETECTION
import json


def run_yolo_world(args):
    """Configures and runs the Ultralytics YOLO-World detector."""
    detector, class_map = YoloWorldDetector.load_detector(args.model, args.prompts)

    return detector.process_directory(
        directory_path=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        score_thresh=args.score_threshold,
        batch_size=args.batch_size,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
    )


def train_yolo_world(args):
    """Configures and trains the Ultralytics YOLO-World detector."""
    print("Initializing Ultralytics YOLO-World detector...")
    detector, _ = YoloWorldDetector.load_model(args.model, args.prompts)

    return detector.train(args)


def add_yolo_world_parser(subparsers, parent_parser, train=False):
    yolo_parser = subparsers.add_parser('yolo_world', help='Use Ultralytics YOLO-World model.', parents=[parent_parser])
    
    if train:
        yolo_parser.add_argument('--yaml-path', type=str, help='Path to the Ultralytics dataset configuration YAML.')
    
    yolo_parser.add_argument('--model', type=str, default='YOLO-X-WORLD', choices=YOLO_WORLD_MODELS.keys())
    yolo_parser.add_argument('--score-threshold', '-t', type=float, default=0.05, help='Detection confidence threshold.')
    yolo_parser.add_argument('--prompts', '-p', nargs='+', required=True, help='Descriptive text prompts for detection.')
    
    if train:
        yolo_parser.set_defaults(func=train_yolo_world)
    else:
        yolo_parser.set_defaults(func=run_yolo_world)


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
        score_thresh = kwargs.get('score_thresh', 0.05)

        results_batch = self.model.predict(images, conf=score_thresh, verbose=False)

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
    def load_detector(model_key, prompts):
        print("Loading Ultralytics YOLO-World detector...")
        model_id = YOLO_WORLD_MODELS[model_key]
        class_map = {k: 0 for k in prompts}
        detector = YoloWorldDetector(model_id)
        
        print(f"Setting model classes to: {list(class_map.keys())}")
        detector.model.set_classes(list(class_map.keys()))
        return detector, class_map

