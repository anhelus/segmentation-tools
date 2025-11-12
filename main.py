import argparse
from pathlib import Path
from src.models.utils import parse_cfg
from src.models.gdino import add_dino_parser
from src.models.owl import add_owl_parser
from src.models.yolo_world import add_yolo_world_parser
from src.models.grounded_sam import add_grounded_sam_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot object detection on a directory of images.")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        'config_path', type=str, help='Path to the dataset configuration YAML.'
    )
    parent_parser.add_argument(
        '--save-predictions-only', action="store_true", default=False,
        help='Wether to save only the predictions without computing metrics.'
    )
    parent_parser.add_argument(
        '--save-metrics-only', action="store_true", default=False,
        help='Wether to save only the metrics without saving predictions.'
    )
    parent_parser.add_argument(
        '--output-name', type=str, default=None,
        help='The name of the output directory. By default, uses the model name along with the current timestamp.'
    )

    subparsers = parser.add_subparsers(dest='model_type', required=True, help='Select the model to run.')
    add_dino_parser(subparsers, parent_parser)
    add_owl_parser(subparsers, parent_parser)
    add_yolo_world_parser(subparsers, parent_parser)
    add_grounded_sam_parser(subparsers, parent_parser)

    args = parser.parse_args()

    config = parse_cfg(args.config_path)
    args.__dict__.update(config)
    
    detector, class_map = args.load_func(args)
    output_root = args.func(detector, class_map, args)

    if output_root:
        classes_file_path = Path(output_root) / 'classes.txt'
        with open(classes_file_path, 'w') as f:
            for class_name in args.class_names:
                f.write(class_name.strip() + '\n')
        print(f"Saved class name to {classes_file_path}")
    else:
        print("\nProcessing finished without saving the labels.")
