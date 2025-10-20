from abc import ABC, abstractmethod
from src.metrics.utils import bbox_iou, segmentation_iou, polygon_to_mask


class EvaluationStrategy(ABC):
    """Abstract base class for an evaluation strategy."""
    @abstractmethod
    def prepare_item(self, item, img_dims):
        """Prepares a single GT or prediction item for comparison."""
        pass

    @abstractmethod
    def calculate_overlap(self, prepared_gt, prepared_pred):
        """Calculates the overlap (IoU) between two prepared items."""
        pass

class BboxStrategy(EvaluationStrategy):
    """Evaluation strategy for bounding boxes."""
    def prepare_item(self, item, img_dims=None):
        return item[1:5]  # Return [x1, y1, x2, y2]

    def calculate_overlap(self, prepared_gt, prepared_pred):
        return bbox_iou(prepared_gt, prepared_pred)

class SegmentationStrategy(EvaluationStrategy):
    """Evaluation strategy for segmentations."""
    def prepare_item(self, item, img_dims):
        # Prediction format: [class, x1, y1,..., conf]
        # Ground Truth format: [class, x1, y1,...]
        # The polygon is all elements between the class_id and the potential confidence score.
        
        is_prediction = len(item) % 2 == 0
        polygon = item[1:-1] if is_prediction else item[1:]
        return polygon_to_mask(polygon, img_dims)

    def calculate_overlap(self, prepared_gt, prepared_pred):
        return segmentation_iou(prepared_gt, prepared_pred)
