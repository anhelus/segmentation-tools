from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
from src.models.base_model import BaseModel
from src.models.constants import OWL_MODELS, DETECTION
import torch


def pred_owl(detector, class_map, args):
    return detector.process_directory(
        directory_path=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        score_threshold=args.score_threshold,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
        batch_size=args.batch_size,
    )


def add_owl_parser(subparsers, parent_parser, train=False, optim=False):
    owl_parser = subparsers.add_parser('owl', help='Use OwlViT model.', parents=[parent_parser])
    owl_parser.add_argument('--model', type=str, default='OWLVIT-BASE-32', choices=OWL_MODELS.keys())
    owl_parser.add_argument('--prompts', nargs='+', required=True, help='Descriptive text prompts for detection.')
    
    if not optim:
        owl_parser.add_argument('--score-threshold', type=float, default=0.1, help='Detection confidence threshold.')

    owl_parser.set_defaults(load_func=OwlDetector.load_detector)

    if not train:
        owl_parser.set_defaults(func=pred_owl)


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
        score_threshold = kwargs.get('score_threshold')

        if score_threshold is None:
            print("Argument score_threshold not specified. Using default value (0.1)")
            score_threshold = 0.1

        prompts = [text_prompts] * n_img

        inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width) for image in images]).to(self.device)
        
        results_raw = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
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

    def train(self, **kwargs):
        pass

    @staticmethod
    def load_detector(args):
        print("Loading OWL-ViT detector...")
        model_id = OWL_MODELS[args.model]
        class_map = {k: 0 for k in args.prompts}
        return OwlDetector(model_id), class_map
