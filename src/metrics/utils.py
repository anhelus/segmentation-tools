import numpy as np
import cv2


def calculate_iou(intersection, area1, area2):
    """
    Generic IoU calculation based on intersection and individual areas.
    """
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


def polygon_to_mask(polygon, img_dims):
    """Converts YOLO-format polygon coordinates to a binary mask."""
    h, w = img_dims
    mask = np.zeros((h, w), dtype=np.uint8)
    points = (np.array(polygon).reshape(-1, 2) * np.array([w, h])).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def bbox_iou(box1_xyxy, box2_xyxy):
    """Calculates IoU for two bounding boxes in [xmin, ymin, xmax, ymax] format."""
    x1_b1, y1_b1, x2_b1, y2_b1 = box1_xyxy
    x1_b2, y1_b2, x2_b2, y2_b2 = box2_xyxy

    inter_x1 = max(x1_b1, x1_b2)
    inter_y1 = max(y1_b1, y1_b2)
    inter_x2 = min(x2_b1, x2_b2)
    inter_y2 = min(y2_b1, y2_b2)

    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2_b1 - x1_b1) * (y2_b1 - y1_b1)
    box2_area = (x2_b2 - x1_b2) * (y2_b2 - y1_b2)
    
    return calculate_iou(intersection_area, box1_area, box2_area)


def segmentation_iou(gt_mask, pred_mask):
    """Calculates IoU for two binary masks."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    area1 = gt_mask.sum()
    area2 = pred_mask.sum()
    return calculate_iou(intersection, area1, area2)


def dice_score(gt_mask, pred_mask):
    """Calculates the Dice Score for two binary masks."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    denominator = gt_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def calculate_map(ground_truths, predictions, iou_threshold, img_dims, strategy):
    """
    Calculates mAP and per-class AP using a provided evaluation strategy.
    Returns:
        tuple: (mean_ap, per_class_ap_dict)
    """
    per_class_ap = {}
    all_classes = sorted(list(set([gt[0] for gt in ground_truths] + [pred[0] for pred in predictions])))

    if not all_classes:
        return 0.0, {}

    for c in all_classes:
        class_gts = [gt for gt in ground_truths if gt[0] == c]
        class_preds = [pred for pred in predictions if pred[0] == c]

        if not class_gts:
            continue
        if not class_preds:
            per_class_ap[c] = 0.0
            continue
        
        class_preds.sort(key=lambda x: x[-1], reverse=True)
        prepared_gts = [strategy.prepare_item(gt, img_dims) for gt in class_gts]
        gt_matched = [False] * len(class_gts)
        
        true_positives = 0
        false_positives = 0
        recalls = []
        precisions = []

        for pred in class_preds:
            prepared_pred = strategy.prepare_item(pred, img_dims)
            best_iou = -1
            best_gt_idx = -1
            
            for gt_i, prepared_gt in enumerate(prepared_gts):
                iou_ = strategy.calculate_overlap(prepared_gt, prepared_pred)
                if iou_ > best_iou:
                    best_iou = iou_
                    best_gt_idx = gt_i
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                true_positives += 1
                gt_matched[best_gt_idx] = True
            else:
                false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / len(class_gts)
            precisions.append(precision)
            recalls.append(recall)

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        # Store AP in the dictionary with class_id as key
        per_class_ap[c] = ap
        
    if not per_class_ap:
        return 0.0, {}
        
    # Calculate mean AP from the dictionary values
    mean_ap = sum(per_class_ap.values()) / len(per_class_ap)
    # Return both the mean and the dictionary
    return mean_ap


def calculate_precision_recall(ground_truths, predictions, iou_threshold, img_dims, strategy):
    """
    Calculates Precision and Recall for a given IoU and confidence threshold.
    Returns:
        dict: {class_id: {'precision': float, 'recall': float}}
    """
    results = {}
    all_classes = sorted(list(set([gt[0] for gt in ground_truths] + [pred[0] for pred in predictions])))

    for c in all_classes:
        class_gts = [gt for gt in ground_truths if gt[0] == c]
        class_preds = [pred for pred in predictions if pred[0] == c]

        num_gts = len(class_gts)
        if num_gts == 0: # No ground truths for this class
            tp = 0
            fp = len(class_preds)
            fn = 0
        else:
            if not class_preds: # No predictions for this class
                tp, fp, fn = 0, 0, num_gts
            else:
                prepared_gts = [strategy.prepare_item(gt, img_dims) for gt in class_gts]
                gt_matched = [False] * len(class_gts)
                tp = 0
                fp = 0
                for pred in class_preds:
                    prepared_pred = strategy.prepare_item(pred, img_dims)
                    best_iou = -1
                    best_gt_idx = -1
                    for gt_i, prepared_gt in enumerate(prepared_gts):
                        iou_ = strategy.calculate_overlap(prepared_gt, prepared_pred)
                        if iou_ > best_iou:
                            best_iou = iou_
                            best_gt_idx = gt_i
                    
                    if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                        tp += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        fp += 1
                fn = num_gts - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / num_gts if num_gts > 0 else 0.0
        results[c] = {'precision': precision, 'recall': recall}

    return results

