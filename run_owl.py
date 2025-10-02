# python .\run_owl.py -p "a photo of a lettuce" "a head of lettuce" "a lettuce plant" "green lettuce leaves" "romaine lettuce" "iceberg letuce" "salad greens" "a green leafy vegetable" -l "lettuce" "lettuce" "lettuce" "lettuce" "lettuce" "lettuce" "lettuce" "lettuce"
import argparse
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from src import owl, constants


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model to use for detection',
        default='OWLVIT-BASE')
    parser.add_argument(
        '-d',
        '--datapath',
        help='Path of the dataset',
        default='dataset')
    parser.add_argument(
        '-t',
        '--threshold',
        help='Detection threshold',
        type=float,
        default=0.07)
    parser.add_argument(
        '-p',
        '--prompts',
        nargs='+',
        help='Prompts to be used for the CLIP model',
        required=True)
    parser.add_argument(
        '-l',
        '--labels',
        nargs='+',
        help='Labels used for the classes')
    args = parser.parse_args()

    model_id = constants.OWL_MODELS[args.model]
    class_map = {k: 0 for k in args.prompts}
    if args.labels is None:
        labels = class_map.keys()
    else:
        labels = args.labels

    # --- Model Loading ---
    print(f"Using device: {device}")
    print(f"Loading model and processor...")
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)
    print("Model and processor loaded.")

    # --- Image Processing ---
    owl.process_images_in_directory(
        directory_path=args.datapath,
        class_map=class_map,
        model=model,
        processor=processor,
        device=device,
        score_thresh=args.threshold,
        out_prompts=labels)

    print("\nProcessing complete.")