import json
import argparse
import numpy as np
from itertools import product
from src.models.utils import parse_cfg
from src.models.gdino import add_dino_parser
from src.models.owl import add_owl_parser
from src.models.yolo_world import add_yolo_world_parser


def param_grid(grid):
    """Generates all combinations of parameters from the grid."""
    keys = grid.keys()
    values = (grid[key] for key in keys)
    for instance in product(*values):
        yield dict(zip(keys, instance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize thresholds for object detection models.")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        'model_type', type=str, choices=["dino", "owl", "yolo_world"], help='The type of model for which to optimize the thresholds.'
    )
    parent_parser.add_argument(
        'config_path', type=str, help='Path to the dataset configuration YAML.'
    )

    subparsers = parser.add_subparsers(dest='model_type', required=True, help='Select the model to optimize.')
    add_dino_parser(subparsers, parent_parser, optim=True)
    add_owl_parser(subparsers, parent_parser, optim=True)
    add_yolo_world_parser(subparsers, parent_parser, optim=True)

    args = parser.parse_args()

    config = parse_cfg(args.config_path)
    args.__dict__.update(config)

    if args.model_type == "dino":
        param_candidates = {
            "box_threshold": [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.65, 0.8],
            "text_threshold": [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.6]
        }
    elif args.model_type in ("owl", "yolo_world"):
        param_candidates = {
            "score_threshold": [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.65, 0.8]
        }
    else:
        raise ValueError(f"Unsupported model type for optimization: {args.model_type}")
    
    best_map = -np.inf
    best_params = None

    for params in param_grid(param_candidates):
        args.__dict__.update(params)

        print(f"Running optimization with parameters: {params}")

        out_root = args.func(args)

        with open(out_root / "metrics.json", 'r') as f:
            eval = json.load(f)

        current_map = eval.get("mAP@75", -np.inf)

        if current_map > best_map:
            best_map = current_map
            best_params = params
        
    print(f"Best parameters: {best_params} with mAP@75: {best_map}")

    