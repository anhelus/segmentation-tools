from abc import ABC, abstractmethod
import torch
from pathlib import Path
from PIL import Image
from src.models import utils
from tqdm import tqdm
from src.models.constants import DETECTION, SEGMENTATION
from src.metrics import BboxMetrics, SegmentationMetrics
import os
import time


class BaseModel(ABC):
    """
    Abstract base class for a zero-shot object detector.
    """
    def __init__(self, model_id):
        self.metrics = BboxMetrics if self.model_type == DETECTION else SegmentationMetrics
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.processor = self.load_model(model_id)
        print("Model and processor loaded.")


    @abstractmethod
    def predict(self, images, class_map, **kwargs):
        """Performs inference on a batch of images."""
        pass
    

    @abstractmethod
    def train(self, **kwargs):
        """Trains the model. To be implemented in subclasses."""
        pass


    def model_identifier(self):
        return self.model_id.split('/')[-1].replace(".pt", "")


    def __setup_prediction(self, input_root, output_name, save_pred):
        if not output_name:
            output_name = self.model_identifier() + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

        output_root = input_root.parent / output_name
        output_root.mkdir(exist_ok=True)
        
        if save_pred:
            try:
                os.symlink(input_root / "images", output_root / "images", target_is_directory=True)
            except FileExistsError:
                pass

        image_paths = utils.find_image_paths(input_root / "images")
        if not image_paths:
            print(f"No images found in {input_root}. Exiting.")
            return None
        
        print(f"Found {len(image_paths)} images.")

        return output_root, image_paths


    def process_directory(self, input_root, model_name, class_map, batch_size=8, **kwargs):
        """Processes all images in a directory, applying the detection model."""
        metrics_only = kwargs["metrics_only"]
        pred_only = kwargs["pred_only"]
        pc_boxes_root = kwargs.get("precomputed_boxes")
        pc_boxes = pc_boxes_root is not None

        if metrics_only and pred_only:
            print("Error: Cannot set both 'metrics_only' and 'pred_only' to True.")
            return None
        
        save_metrics = not pred_only
        save_pred = not metrics_only
        
        output_name = kwargs.get('output_name')
        output_root, image_paths = self.__setup_prediction(input_root, output_name, save_pred)
        
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        elapsed_times = []

        if save_metrics or save_pred:
            prediction_data = {
                "images": [],
                "results": []
            }
        
        print(f"Starting processing with batch size {batch_size}...")

        if pc_boxes:
            pc_boxes_dir = pc_boxes_root / "labels"

        for i in tqdm(range(num_batches), desc=f"Processing batches for {model_name}"):
            batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            predict_kwargs = kwargs.copy()

            if pc_boxes:
                predict_kwargs['precomputed_boxes'] = utils.load_precomputed_boxes_batch(pc_boxes_dir, batch_paths, batch_images)

            start_time = time.time()
            batch_results = self.predict(batch_images, class_map, **predict_kwargs)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)
            
            if save_pred:
                prediction_data["images"].extend(batch_images)
            if save_pred or save_metrics:
                prediction_data["results"].extend(batch_results)
        
        avg_time = sum(elapsed_times) / len(elapsed_times)
        print(f"Average inference time per batch: {avg_time:.2f} seconds")

        if save_pred:
            utils.save_labels(prediction_data["images"], image_paths, prediction_data["results"], output_root)

        if save_metrics:
            img_dims=kwargs.get('image_size', [720, 1280])

            print("Loading ground truth data for evaluation...")
            ground_truths = utils.load_yolo_gt(input_root, img_dims)

            metadata = {
                "batch_size": batch_size,
                "average_inference_time": avg_time,
                **kwargs
            }

            map_thresh_list = kwargs.get('map_thresh_list', [0.5, 0.75])
            utils.save_eval(self, ground_truths, prediction_data, img_dims, map_thresh_list, output_root, metadata)

        return output_root
    
