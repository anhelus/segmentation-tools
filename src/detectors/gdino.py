from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.detectors.base_detector import BaseDetector
from src.metrics import BboxMetrics
import torch

class GDinoDetector(BaseDetector):
    def __init__(self, model_id):
        self.metrics = BboxMetrics
        super().__init__(model_id)

    """
    Detector class for the Grounding DINO model.
    """
    def load_model(self, model_id):
        """
        Loads the Grounding DINO model and processor.
        """
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        return model, processor

    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Grounding DINO model.
        """
        text_for_model = list(class_map.keys())
        text_threshold = kwargs.get('text_threshold', 0.3)
        box_threshold = kwargs.get('box_threshold', 0.35)

        # The text input is a list of lists, one for each image in the batch
        text_inputs = [text_for_model] * len(images)

        inputs = self.processor(
            images=images,
            text=text_inputs,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Prepare target sizes for post-processing
        target_sizes = [image.size[::-1] for image in images]
        
        results_raw = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=target_sizes,
            threshold=box_threshold,
            text_threshold=text_threshold
        )

        all_processed_results = []
        for i, result in enumerate(results_raw):
            processed_for_image = []
            for score, label, box in zip(result["scores"], result["text_labels"], result["boxes"]):
                if label in class_map:
                    processed_for_image.append({
                        "score": score.item(),
                        "label": label,
                        "box": box.tolist(),
                        "class_index": class_map[label]
                    })
            all_processed_results.append(processed_for_image)
            
        return all_processed_results

