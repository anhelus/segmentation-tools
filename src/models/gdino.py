from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.models.base_model import BaseModel
from src.models.constants import DINO_MODELS, DETECTION
import torch


def run_dino(args):
    """Configures and runs the Grounding DINO detector."""
    print("Initializing Grounding DINO detector...")
    model_id = DINO_MODELS[args.model]
    class_map = {label: idx for idx, label in enumerate(args.class_names)}
    detector = GDinoDetector(model_id)

    return detector.process_directory(
        directory_path=args.bbox_gt,
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


def train_dino(args):
    pass


def add_dino_parser(subparsers, parent_parser, train=False):
    dino_parser = subparsers.add_parser('dino', help='Use Grounding DINO model.', parents=[parent_parser])
    dino_parser.add_argument('--model', type=str, default='GDINO-BASE', choices=DINO_MODELS.keys())
    dino_parser.add_argument('--text-threshold', type=float, default=0.25, help='Confidence threshold for text matching.')
    dino_parser.add_argument('--box-threshold', '-t', type=float, default=0.2, help='Confidence threshold for object detection box.')
    
    if train:
        dino_parser.set_defaults(func=train_dino)
    else:
        dino_parser.set_defaults(func=run_dino)


class GDinoDetector(BaseModel):
    def __init__(self, model_id):
        self.model_type = DETECTION
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
        box_threshold = kwargs.get('threshold', 0.35)

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

    def train(self, **kwargs):
        pass
