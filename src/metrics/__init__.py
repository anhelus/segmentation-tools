from abc import ABC, abstractmethod
from src.metrics.utils import calculate_precision_recall, calculate_map, segmentation_iou, dice_score   
from src.metrics.strategies import EvaluationStrategy, BboxStrategy, SegmentationStrategy


class EvalMetrics(ABC):
    strategy: EvaluationStrategy

    @staticmethod
    @abstractmethod
    def evaluate(ground_truths, predictions, img_dims, thresh_list):
        pass


class BboxMetrics(EvalMetrics):
    strategy = BboxStrategy()
    
    @staticmethod
    def evaluate(ground_truths, predictions, img_dims, thresh_list: list = [0.5, 0.75]):
        n_gt = len(ground_truths)
        eval = {}
        for thresh in thresh_list:
            disp_thresh = int(thresh*100)
            tot_p = tot_r = tot_map = 0
            for img_gt, img_pred in zip(ground_truths, predictions):
                pr = calculate_precision_recall(img_gt, img_pred, thresh, img_dims, BboxMetrics.strategy)
                tot_p += pr[0]['precision']
                tot_r += pr[0]['recall']
                tot_map += calculate_map(img_gt, img_pred, thresh, img_dims, BboxMetrics.strategy)

            eval[f'Precision@{disp_thresh}'] = tot_p / n_gt
            eval[f'Recall@{disp_thresh}'] = tot_r / n_gt
            eval[f'mAP@{disp_thresh}'] = tot_map / n_gt   
        
        return eval


class SegmentationMetrics(EvalMetrics):
    strategy = SegmentationStrategy()
    
    @staticmethod
    def evaluate(ground_truths, predictions, img_dims, thresh_list: list = [0.5, 0.75]):
        n_gt = len(ground_truths)
        eval = {}
        
        for thresh in thresh_list:
            disp_thresh = int(thresh*100)
            tot_p = tot_r = tot_map = 0
            
            for img_gt, img_pred in zip(ground_truths, predictions):
                pr = calculate_precision_recall(img_gt, img_pred, thresh, img_dims, SegmentationMetrics.strategy)
                tot_p += pr[0]['precision']  # TODO adapt to multiclass
                tot_r += pr[0]['recall']
                tot_map += calculate_map(img_gt, img_pred, thresh, img_dims, SegmentationMetrics.strategy)
            
            eval[f'Precision@{disp_thresh}'] = tot_p / n_gt
            eval[f'Recall@{disp_thresh}'] = tot_r / n_gt
            eval[f'mAP@{disp_thresh}'] = tot_map / n_gt
        
        tot_iou = tot_dice = 0
        for img_gt, img_pred in zip(ground_truths, predictions):
            tot_iou += segmentation_iou(ground_truths, predictions)
            tot_dice += dice_score(ground_truths, predictions)
        
        eval['IoU'] = tot_iou / n_gt   
        eval['Dice'] = tot_dice / n_gt

        return eval


if __name__ == "__main__":
    IMAGE_DIMENSIONS = (640, 640) # (height, width)

    print("--- 📦 Bounding Box Metrics ---")
    IOU_THRESHOLD_BBOX = 0.50

    ground_truths_bbox = [
        [0, 0.4, 0.4, 0.6, 0.6],
        [1, 0.125, 0.175, 0.275, 0.425],
        [1, 0.7, 0.55, 0.9, 0.85]
    ]
    predictions_bbox = [
        [0, 0.41, 0.42, 0.61, 0.62, 0.95],
        [1, 0.13, 0.18, 0.28, 0.43, 0.88],
        [1, 0.72, 0.56, 0.89, 0.84, 0.92],
        [0, 0.15, 0.75, 0.25, 0.85, 0.70],
        [1, 0.45, 0.45, 0.55, 0.55, 0.65] # Below CONF_THRESHOLD
    ]
    
    bbox_strategy = BboxStrategy()
    
    map_bbox, ap_bbox = calculate_map(ground_truths_bbox, predictions_bbox, IOU_THRESHOLD_BBOX, IMAGE_DIMENSIONS, bbox_strategy)
    print(f"📈 mAP@{IOU_THRESHOLD_BBOX:.2f}: {map_bbox:.4f}")
    for class_id, ap in ap_bbox.items():
        print(f"   - Class {class_id} AP: {ap:.4f}")
    
    pr_bbox = calculate_precision_recall(ground_truths_bbox, predictions_bbox, IOU_THRESHOLD_BBOX, IMAGE_DIMENSIONS, bbox_strategy)
    print(f"\n🎯 Precision & Recall (@{IOU_THRESHOLD_BBOX:.2f} IoU:")
    total_precision, total_recall = 0, 0
    for class_id, metrics in pr_bbox.items():
        print(f"   - Class {class_id}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        total_precision += metrics['precision']
        total_recall += metrics['recall']
    if pr_bbox:
        print(f"   - Macro Avg: Precision={total_precision/len(pr_bbox):.4f}, Recall={total_recall/len(pr_bbox):.4f}")

    print("-" * 35)

    print("\n--- 🎨 Segmentation Metrics ---")
    IOU_THRESHOLD_SEG = 0.75

    ground_truths_seg = [
        [0, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6],
        [1, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]
    ]
    predictions_seg = [
        [0, 0.41, 0.41, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6, 0.95],
        [1, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4, 0.88],
        [0, 0.8, 0.8, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9, 0.70]
    ]
    
    segmentation_strategy = SegmentationStrategy()

    # Calculate and display mAP and per-class AP
    map_seg, ap_seg = calculate_map(ground_truths_seg, predictions_seg, IOU_THRESHOLD_SEG, IMAGE_DIMENSIONS, segmentation_strategy)
    print(f"📈 mAP@{IOU_THRESHOLD_SEG:.2f}: {map_seg:.4f}")
    for class_id, ap in ap_seg.items():
        print(f"   - Class {class_id} AP: {ap:.4f}")
        
    # Calculate and display Precision and Recall
    pr_seg = calculate_precision_recall(ground_truths_seg, predictions_seg, IOU_THRESHOLD_SEG, IMAGE_DIMENSIONS, segmentation_strategy)
    print(f"\n🎯 Precision & Recall (@{IOU_THRESHOLD_SEG:.2f} IoU:")
    total_precision, total_recall = 0, 0
    for class_id, metrics in pr_seg.items():
        print(f"   - Class {class_id}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        total_precision += metrics['precision']
        total_recall += metrics['recall']
    if pr_seg:
        print(f"   - Macro Avg: Precision={total_precision/len(pr_seg):.4f}, Recall={total_recall/len(pr_seg):.4f}")
    
    # Calculate and display a sample Dice Score
    gt_mask_sample = segmentation_strategy.prepare_item(ground_truths_seg[0], IMAGE_DIMENSIONS)
    pred_mask_sample = segmentation_strategy.prepare_item(predictions_seg[0], IMAGE_DIMENSIONS)
    dice = dice_score(gt_mask_sample, pred_mask_sample)
    print(f"\n🎲 Sample Dice Score (Class 0 GT vs. Pred 0): {dice:.4f}")
