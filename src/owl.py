import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
from src.utils import convert_to_yolo_format

def draw_and_save_bounding_boxes(image, results, output_path, class_list):
    """
    Draws bounding boxes, labels, and scores on an image and saves it.
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    colors = {cls: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 200)) for cls in class_list}
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    result = results[0]
    boxes, scores, text_labels = result["boxes"], result["scores"], class_list

    for box, score, text_label in zip(boxes, scores, text_labels):
        box = [round(i, 2) for i in box.tolist()]
        
        box_color = colors.get(text_label, "red")
        
        draw.rectangle(box, outline=box_color, width=3)
        
        text = f"{text_label}: {score:.2f}"
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_location = [box[0], box[1] - text_height - 2]
        if text_location[1] < 0:
            text_location[1] = box[1] + 2
            
        draw.rectangle(
            (text_location[0], text_location[1], text_location[0] + text_width + 4, text_location[1] + text_height),
            fill=box_color
        )
        draw.text((text_location[0] + 2, text_location[1]), text, fill="white", font=font)

    img_with_boxes.save(output_path)


def process_images_in_directory(
        directory_path,
        class_map,
        model,
        processor,
        device,
        score_thresh,
        out_prompts=None):
    """
    Recursively processes images using OwlViT to find specific objects.
    """
    p = Path(directory_path)
    if not p.is_dir():
        print(f"Error: Directory not found at {p}")
        return

    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = [file for ext in image_extensions for file in p.rglob(ext)]

    print(f"Found {len(image_paths)} images in {directory_path}. Starting processing...")

    text_prompts = list(class_map.keys())
    text_queries = [text_prompts]
    if out_prompts == None:
        out_prompts = text_prompts

    for image_path in image_paths:
        print(f"\n--- Processing: {image_path.name} ---")
        try:
            image = Image.open(image_path).convert("RGB")
            
            inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([(image.height, image.width)]).to(device)
            
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=score_thresh,
                text_labels=text_queries 
            )
            
            result_for_image = results[0]
            boxes, scores, detected_labels = result_for_image["boxes"], result_for_image["scores"], result_for_image["text_labels"]

            if len(scores) == 0:
                print(f"DEBUG: No objects detected at the current threshold of {score_thresh}.")
            else:
                print("DEBUG: Raw detections (before filtering by your class map):")
                for score, label in zip(scores, detected_labels):
                    print(f"  - Found '{label}' with confidence {score:.3f}")

            yolo_output_path = Path(image_path.parent, 'labels')
            yolo_output_path.mkdir(parents=True, exist_ok=True)
            yolo_output_path = Path(yolo_output_path, f'{image_path.stem}.txt')
            found_lettuces = 0
            with open(yolo_output_path, "w") as out_file:
                for score, label, box in zip(scores, detected_labels, boxes):
                    if label in class_map:
                        class_index = class_map[label]
                        yolo_box = convert_to_yolo_format(box.tolist(), image.size)
                        out_file.write(f"{class_index} {' '.join(f'{coord:.6f}' for coord in yolo_box)}\n")
                        print(f"SUCCESS: Matched '{label}' to class index {class_index} with confidence {score:.3f}")
                        found_lettuces += 1

            # --- 2. Move image ---
            image_output_path = Path(image_path.parent, 'images')
            image_output_path.mkdir(parents=True, exist_ok=True)
            image_path.rename(Path(image_output_path, image_path.name))

            if found_lettuces == 0:
                print("INFO: No lettuces found that matched the provided prompts and thresholds in this image.")


            # --- 3. Save image with bounding boxes drawn ---
            bb_image_path = Path(image_path.parent, 'labeled')
            bb_image_path.mkdir(parents=True, exist_ok=True) 
            draw_and_save_bounding_boxes(image, results, Path(bb_image_path, image_path.name), out_prompts)
            print(f"Saved annotated image to {bb_image_path.name}")

        except Exception as e:
            print(f"ERROR: Could not process file {image_path}: {e}")