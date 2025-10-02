# usage: python run_gdino -l lettuce
import argparse
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src import gdino, constants


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model to use for detection',
        default='GDINO-TINY')
    parser.add_argument(
        '-d',
        '--datapath',
        help='Path of the dataset',
        default='dataset')
    parser.add_argument(
        '-l',
        '--labels',
        nargs='+',
        help='Target classes',
        required=True)
    args = parser.parse_args()

    model_id = constants.DINO_MODELS[args.model]

    text_labels = [(label, idx) for idx, label in enumerate(args.labels)]

    # --- Model Loading ---
    print(f"Using device: {device}")
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("Model and processor loaded.")

    # --- Image Processing ---
    gdino.process_images_in_directory(
        args.datapath,
        text_labels,
        model,
        processor,
        device)

    print("\nProcessing complete.")