from ultralytics import YOLOWorld
from src.detectors.base_detector import BaseDetector
from src.metrics import BboxMetrics


class YoloWorldDetector(BaseDetector):
    """
    Detector class for the YOLO-World model from the Ultralytics library.
    """
    def __init__(self, model_id):
        self.metrics = BboxMetrics
        super().__init__(model_id)
    
    
    def load_model(self, model_id):
        """
        Loads the YOLOWorld model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        model = YOLOWorld(model_id)
        # No separate processor object is needed for this implementation
        return model, None

    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Ultralytics YOLO-World model.
        The model's classes should be set once before calling this method.
        """
        score_thresh = kwargs.get('score_thresh', 0.05)

        # The model.predict method can take a list of PIL images directly
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

