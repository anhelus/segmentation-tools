from pathlib import Path
import shutil
import cv2
import yaml
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def find_image_paths(directory_path):
    """Finds all image files in a directory recursively."""
    p = Path(directory_path)
    if not p.is_dir():
        print(f"Error: Directory not found at {p}")
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    return [file for ext in image_extensions for file in p.rglob(ext)]


def parse_cfg(yaml_file_path):
    """
    Parses a YAML file and extracts the absolute paths for specified keys.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        base_path = Path(config.get('path'))
        if not base_path:
            raise ValueError("The 'path' key is missing or empty in the YAML file.")

        keys_to_process = ['bbox_gt', 'seg_gt']

        for key in keys_to_process:
            relative_path = config.get(key)
            if relative_path:
                # Join base and relative paths, then get the absolute path
                gt_path = base_path / relative_path
                config[key] = gt_path.resolve()
            else:
                config[key] = None

        return config

    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None


def convert_to_yolo_format(box, image_size):
    """
    Converts a bounding box from [x_min, y_min, x_max, y_max] to YOLO format.
    """
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = box

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return x_center_normalized, y_center_normalized, width_normalized, height_normalized


def yolo_to_xyxy(yolo_box, image_width, image_height):
    """Converts a YOLO format bounding box to [xmin, ymin, xmax, ymax] format."""
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_box
    
    box_width = width_norm * image_width
    box_height = height_norm * image_height
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    
    xmin = x_center - (box_width / 2)
    ymin = y_center - (box_height / 2)
    xmax = x_center + (box_width / 2)
    ymax = y_center + (box_height / 2)
    
    return [xmin, ymin, xmax, ymax]


def load_precomputed_boxes(label_path, image_width, image_height):
    """Loads bounding boxes from a YOLO format .txt file."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5: # class_id x_center y_center width height
                yolo_coords = [float(p) for p in parts[1:]]
                xyxy_box = yolo_to_xyxy(yolo_coords, image_width, image_height)
                boxes.append(xyxy_box)
    return boxes


def draw_boxes(image, results, output_path):
    """Draws bounding boxes on an image and saves it."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        box = result['box']
        label = result['label']
        score = result['score']
        
        color = tuple(np.random.randint(0, 256, 3))
        draw.rectangle(box, outline=color, width=3)
        
        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), text, fill="white", font=font)

    img_with_boxes.save(output_path)


def mask_to_yolo_segmentation(mask, image_width, image_height):
    """
    Converts a binary mask to a YOLO segmentation format string.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    
    contour = max(contours, key=cv2.contourArea)
    
    if contour.ndim > 2:
        contour = contour.squeeze()

    normalized_contour = contour.astype(np.float32)
    normalized_contour[:, 0] /= image_width
    normalized_contour[:, 1] /= image_height
    
    return " ".join(f"{coord:.6f}" for point in normalized_contour for coord in point)


def draw_boxes_and_masks(image, results, output_path):
    """
    Draws both bounding boxes and segmentation masks on an image and saves it.
    """
    image_np = np.array(image.convert("RGB"))
    overlay = image_np.copy()
    
    label_colors = {}

    for result in results:
        label = result['label']
        if label not in label_colors:
            label_colors[label] = np.random.randint(0, 256, size=3).tolist()
        
        color = label_colors[label]
        mask = result.get('mask')
        
        if mask is not None:
            colored_mask = np.zeros_like(image_np, dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(colored_mask, contours, -1, color, -1)
            cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0, overlay)

    final_image = cv2.addWeighted(image_np, 0.5, overlay, 0.5, 0)

    for result in results:
        label = result['label']
        score = result['score']
        mask = result.get('mask')
        color = label_colors[label]

        box = [int(c) for c in result['box']]

        # Draw mask contour for better visibility
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_image, contours, -1, color, 2)
        
        # Draw the original bounding box that SAM exploited
        cv2.rectangle(final_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label background and text
        text = f"{label}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(final_image, (box[0], box[1] - th - 5), (box[0] + tw, box[1]), color, -1)
        cv2.putText(final_image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    final_pil_image = Image.fromarray(final_image)
    final_pil_image.save(output_path)


def load_yolo_gt(gt_directory, img_dims):
    image_height, image_width = img_dims
    gt_list = []
    labels_dir = gt_directory / "labels"
    for label_file in labels_dir.glob("*.txt"):
        gt = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 5: # class_id x_center y_center width height
                    yolo_class = int(parts[0])
                    yolo_coords = [float(p) for p in parts[1:]]
                    points_xy = yolo_to_xyxy(yolo_coords, image_width, image_height)
                elif len(parts) > 5: # class_id x1 y1 x2 y2 ...
                    yolo_class = int(parts[0])
                    points_xy = [float(p) for p in parts[1:]]

                gt.append([yolo_class, *points_xy])
        gt_list.append(gt)
    return gt_list


def save_labels(images, paths, results, output_root):
    labels_dir = output_root / 'labels'
    labeled_dir = output_root / 'labeled'

    labels_dir.mkdir(exist_ok=True)
    labeled_dir.mkdir(exist_ok=True) # TODO make labeled images saving optional

    for image, path, image_results in tqdm(zip(images, paths, results), total=len(images), desc="Saving labels", unit="img"):
        yolo_output_path = labels_dir / f'{path.stem}.txt'
        labeled_image_path = labeled_dir / path.name
        
        with yolo_output_path.open("w") as out_file:
            if image_results:
                for image_result in image_results:
                    has_segmentation = image_result.get('mask') is not None

                    if has_segmentation:
                        yolo_data = mask_to_yolo_segmentation(image_result['mask'], image.width, image.height)
                    else:
                        yolo_box = convert_to_yolo_format(image_result['box'], image.size)
                        yolo_data = " ".join(f'{c:.6f}' for c in yolo_box)

                    if yolo_data:
                        out_file.write(f"{image_result['class_index']} {yolo_data}\n")
            else:
                shutil.copy(path, labeled_image_path)
                continue

        if image_results[0].get('mask') is not None:
            draw_boxes_and_masks(image, image_results, labeled_image_path)
        else:
            draw_boxes(image, image_results, labeled_image_path)


def save_metrics(eval_metrics, output_root):
    metrics_path = output_root / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)

