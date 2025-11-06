from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
from src.models.base_model import BaseModel
from src.models.constants import DETECTION

import torch

class OwlDetector(BaseModel):
    """
    Detector class for the OwlViT model.
    """
    def __init__(self, model_id):
        self.model_type = DETECTION
        super().__init__(model_id)
    

    def load_model(self, model_id):
        """
        Loads the OwlViT model and processor.
        """
        if model_id.startswith('google/owlv2'):
            processor = Owlv2Processor.from_pretrained(model_id)
            model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        elif model_id.startswith('google/owlvit'):
            processor = OwlViTProcessor.from_pretrained(model_id)
            model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        else:
            raise ValueError(f"Unsupported model_id for OwlDetector: {model_id}")
        
        return model, processor

    def predict(self, images, class_map, **kwargs):
        """
        Performs inference using the OwlViT model.
        """
        n_img = len(images)
        text_prompts = list(class_map.keys())
        score_thresh = kwargs.get('score_thresh', 0.1)

        prompts = [text_prompts] * n_img

        inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width) for image in images]).to(self.device)
        
        results_raw = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_thresh,
            text_labels=prompts
        )
        
        processed_results = []
        for result in results_raw:
            processed_result = []
            for score, label, box in zip(result["scores"], result["text_labels"], result["boxes"]):
                if label in class_map:
                    processed_result.append({
                        "score": score.item(),
                        "label": label,
                        "box": box.tolist(),
                        "class_index": class_map[label]
                    })
            processed_results.append(processed_result)
        return processed_results

