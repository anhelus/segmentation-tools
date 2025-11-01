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
    def model_identifier(self):
        """Returns the name of the model."""
        pass


    def model_identifier(self):
        return self.model_id


    def process_directory(self, directory_path, model_name, class_map, batch_size=8, **kwargs):
        """Processes all images in a directory, applying the detection model."""
        metrics_only = kwargs["metrics_only"]
        pred_only = kwargs["pred_only"]
        
        complete_pipeline = not (metrics_only or pred_only)
        save_metrics = metrics_only or complete_pipeline
        save_pred = pred_only or complete_pipeline

        image_paths = utils.find_image_paths(directory_path)
        if not image_paths:
            print(f"No images found in {directory_path}. Exiting.")
            return None
        
        output_root = directory_path.parent / model_name
        output_root.mkdir(exist_ok=True)
        
        if save_pred:
            try:
                os.symlink(directory_path / "images", output_root / "images", target_is_directory=True)
            except FileExistsError:
                pass
        
        if save_metrics:
            if self.model_type == DETECTION:
                pred_list = lambda pred: [pred["class_index"], *pred["box"], pred["score"]]
            elif self.model_type == SEGMENTATION:
                pred_list = lambda pred: [pred["class_index"], pred["mask"], pred["score"]]
        
        print(f"Found {len(image_paths)} images. Starting processing with batch size {batch_size}...")
        
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        elapsed_times = []
        aggregated_predictions = []

        for i in tqdm(range(num_batches), desc=f"Processing batches for {model_name}"):
            batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            predict_kwargs = kwargs.copy()

            start_time = time.time()
            batch_results = self.predict(batch_images, class_map, **predict_kwargs)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

            if save_metrics:
                for sample_result in batch_results:
                    aggregated_predictions.append([pred_list(prediction) for prediction in sample_result])
            
            if save_pred: # TODO da spostare fuori dal loop per non interferire con il calcolo del tempo
                utils.save_labels(batch_images, batch_paths, batch_results, output_root)
        
        avg_time = sum(elapsed_times) / len(elapsed_times)
        print(f"Average inference time per batch: {avg_time:.2f} seconds")

        if save_metrics:
            img_dims=kwargs.get('image_size', [720, 1280])
            ground_truths = utils.load_yolo_gt(directory_path, img_dims)

            eval = {}
            eval["model_name"] = self.model_identifier()
            eval["batch_size"] = batch_size
            eval.update(kwargs)
            eval["average_inference_time"] = avg_time
            metrics = self.metrics.evaluate(ground_truths, aggregated_predictions, img_dims, thresh_list=kwargs.get('map_thresh_list', [0.5, 0.75]))
            eval.update(metrics)

            utils.save_metrics(eval, output_root)

        return output_root if save_pred else None
    

    def process_precomputed_boxes(self, directory_path, model_name, class_map, batch_size=8, **kwargs):
        """Processes all images in a directory, applying the detection model."""
        image_paths = utils.find_image_paths(directory_path)
        if not image_paths:
            print(f"No images found in {directory_path}. Exiting.")
            return None
        
        output_root = directory_path.parent / "seg_ground_truth"
        output_root.mkdir(exist_ok=True)
        
        os.symlink(directory_path / "images", output_root / "images", target_is_directory=True)
        
        print(f"Found {len(image_paths)} images. Starting processing with batch size {batch_size}...")
        
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        label_dir = directory_path / "labels"
        if not label_dir:
            print("Error: 'label_dir' must be provided in kwargs for precomputed boxes.")
            return None

        for i in tqdm(range(num_batches), desc=f"Processing batches for {model_name}"):
            batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            predict_kwargs = kwargs.copy()
            precomputed_boxes_batch = []
            
            for path, image in zip(batch_paths, batch_images):
                label_file = Path(label_dir) / f'{path.stem}.txt'
                if label_file.exists():
                    boxes = utils.load_precomputed_boxes(label_file, image.width, image.height)
                    precomputed_boxes_batch.append(boxes)
                else:
                    print(f"Warning: Label file not found for {path.name}, will process without precomputed boxes.")
                    precomputed_boxes_batch.append([])

            predict_kwargs['precomputed_boxes'] = precomputed_boxes_batch
            batch_results = self.predict(batch_images, class_map, **predict_kwargs)
            utils.save_labels(batch_images, batch_paths, batch_results, output_root)

        return output_root

