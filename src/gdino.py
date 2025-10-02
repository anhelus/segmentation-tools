import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from pdb import set_trace
from src.utils import convert_to_yolo_format


def draw_and_save_bounding_boxes(
        image,
        results,
        output_path):
    """
    Draws bounding boxes, labels, and scores on an image and saves it.

    Args:
        image (PIL.Image.Image): The original image.
        results (list): The list of detection results from the model.
        output_path (Path): The path to save the new image with drawings.
    """
    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a font, fall back to default if not found
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            
            # Draw the bounding box
            draw.rectangle(box, outline="red", width=3)
            
            # Prepare the text
            text = f"{label}: {score:.2f}"
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw a filled rectangle behind the text for better visibility
            text_location = [box[0], box[1] - text_height]
            if text_location[1] < 0: # If the box is at the top of the image
                text_location[1] = box[1]
                
            draw.rectangle(
                (text_location[0], text_location[1], text_location[0] + text_width, text_location[1] + text_height),
                fill="red"
            )
            # Draw the text
            draw.text(text_location, text, fill="white", font=font)

    # Save the new image
    img_with_boxes.save(output_path)


def process_images_in_directory(
        directory_path,
        text_labels_with_classes,
        model,
        processor,
        device):
    """
    Recursively processes all images in a directory, saves YOLO annotations,
    and saves a new image with bounding boxes drawn.
    """
    p = Path(directory_path)
    if not p.is_dir():
        print(f"Error: Directory not found at {p}")
        return

    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = [file for ext in image_extensions for file in p.rglob(ext)]

    print(f"Found {len(image_paths)} images in {directory_path}. Starting processing...")

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_size = image.size

            text_for_model = [label for label, _ in text_labels_with_classes]

            inputs = processor(images=image, text=[text_for_model], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=0.3,
                target_sizes=[image_size[::-1]]
            )

            # --- 1. Save YOLO annotation file ---
            yolo_output_path = Path(image_path.parent, 'labels')
            yolo_output_path.mkdir(parents=True, exist_ok=True)
            yolo_output_path = Path(yolo_output_path, f'{image_path.stem}.txt')
            with open(yolo_output_path, "w") as out_file:
                for result in results:
                    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                        class_index = -1
                        for lbl, idx in text_labels_with_classes:
                            if label == lbl:
                                class_index = idx
                                break

                        if class_index != -1:
                            yolo_box = convert_to_yolo_format(box.tolist(), image_size)
                            out_file.write(f"{class_index} {' '.join(f'{coord:.6f}' for coord in yolo_box)}\n")
                            print(f"Detected '{label}' in '{image_path.name}' with confidence {score.item():.3f}")
            
            # --- 2. Move image ---
            image_output_path = Path(image_path.parent, 'images')
            image_output_path.mkdir(parents=True, exist_ok=True)
            image_path.rename(Path(image_output_path, image_path.name))

            # --- 3. Save image with bounding boxes drawn ---
            bb_image_path = Path(image_path.parent, 'labeled')
            bb_image_path.mkdir(parents=True, exist_ok=True) 
            draw_and_save_bounding_boxes(image, results, Path(bb_image_path, image_path.name))
            print(f"Saved annotated image to {bb_image_path.name}")


        except Exception as e:
            print(f"Could not process file {image_path}: {e}")