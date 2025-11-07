import argparse
from pathlib import Path
from src.models.utils import parse_cfg
from src.models.gdino import add_dino_parser
from src.models.owl import add_owl_parser
from src.models.yolo_world import add_yolo_world_parser
from src.models.grounded_sam import add_grounded_sam_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run open-world object detection model training.")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        'config_path', type=str, help='Path to the dataset configuration YAML.'
    )
    parent_parser.add_argument(
        '--output-name', type=str, default=None,
        help='The name of the output checkpoint file. By default, uses the model name along with the current timestamp.'
    )

    subparsers = parser.add_subparsers(dest='model_type', required=True, help='Select the model to run.')
    add_dino_parser(subparsers, parent_parser, train=True)
    add_owl_parser(subparsers, parent_parser, train=True)
    add_yolo_world_parser(subparsers, parent_parser, train=True)
    add_grounded_sam_parser(subparsers, parent_parser, train=True)

    args = parser.parse_args()

    config = parse_cfg(args.config_path)
    args.__dict__.update(config)
    
    args.func(args)
