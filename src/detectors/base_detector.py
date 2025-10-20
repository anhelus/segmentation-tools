from abc import ABC, abstractmethod
import torch
from pathlib import Path
from PIL import Image
from src.detectors import utils
from tqdm import tqdm
import os
import time


class BaseDetector(ABC):
    """
    Abstract base class for a zero-shot object detector.
    """
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.processor = self.load_model(model_id)
        print("Model and processor loaded.")


    @abstractmethod
    def load_model(self, model_id):
        """Loads the model and processor from the given model_id."""
        pass


    @abstractmethod
    def predict(self, images, class_map, **kwargs):
        """Performs inference on a batch of images."""
        pass


    def process_bbox_directory(self, directory_path, model_name, class_map, batch_size=8, **kwargs):
        """Processes all images in a directory, applying the detection model."""
        save = kwargs.get("save_results", False)
        image_paths = utils.find_image_paths(directory_path)
        if not image_paths:
            print(f"No images found in {directory_path}. Exiting.")
            return None
        
        if save:
            output_root = directory_path.parent / model_name
            output_root.mkdir(exist_ok=True)

            try:
                os.symlink(directory_path / "images", output_root / "images", target_is_directory=True)
            except FileExistsError:
                pass
        
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

            for sample_result in batch_results:
                aggregated_predictions.append(
                    [[prediction["class_index"], *prediction["box"], prediction["score"]] for prediction in sample_result]
                )
            
            if save:
                utils.save_labels(batch_images, batch_paths, batch_results, output_root)
        
        if not save:
            return None
        
        avg_time = sum(elapsed_times) / len(elapsed_times)
        print(f"Average inference time per batch: {avg_time:.2f} seconds")

        img_dims=kwargs.get('image_size', [1280, 720])
        ground_truths = utils.load_yolo_gt(directory_path, img_dims)

        eval = self.metrics.evaluate(ground_truths, aggregated_predictions, img_dims, thresh_list=kwargs.get('map_thresh_list', [0.5, 0.75]))
        eval["batch_size"] = batch_size
        eval["average_inference_time"] = avg_time

        utils.save_metrics(eval, output_root)

        return output_root
    

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

