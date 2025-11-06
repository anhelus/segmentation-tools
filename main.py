import argparse
from pathlib import Path
from src.models import constants
from src.models.utils import parse_cfg
from src.models.gdino import GDinoDetector
from src.models.owl import OwlDetector
from src.models.yolo_world import YoloWorldDetector
from src.models.grounded_sam import GroundedSamDetector


def run_dino(args):
    """Configures and runs the Grounding DINO detector."""
    print("Initializing Grounding DINO detector...")
    model_id = constants.DINO_MODELS[args.model]
    class_map = {label: idx for idx, label in enumerate(args.class_names)}
    detector = GDinoDetector(model_id)

    return detector.process_directory(
        directory_path=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        batch_size=args.batch_size,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
        image_size=args.image_size,
        map_thresh_list=args.map_thresh,
    )


def run_owl(args):
    """Configures and runs the OwlViT detector."""
    print("Initializing OwlViT detector...")
    model_id = constants.OWL_MODELS[args.model]
    class_map = {k: 0 for k in args.prompts}
    out_labels = args.class_names if args.class_names is not None else list(class_map.keys())
    detector = OwlDetector(model_id)

    return detector.process_directory(
        directory_path=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        score_threshold=args.score_threshold,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
        out_prompts=out_labels,
        batch_size=args.batch_size,
    )


def run_yolo_world(args):
    """Configures and runs the Ultralytics YOLO-World detector."""
    print("Initializing Ultralytics YOLO-World detector...")
    model_id = constants.YOLO_WORLD_MODELS[args.model]
    class_map = {k: 0 for k in args.prompts}
    detector = YoloWorldDetector(model_id)
    
    print(f"Setting model classes to: {list(class_map.keys())}")
    detector.model.set_classes(list(class_map.keys()))

    return detector.process_directory(
        directory_path=args.bbox_gt,
        model_name=args.model_type,
        class_map=class_map,
        score_thresh=args.score_threshold,
        batch_size=args.batch_size,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
    )


def run_grounded_sam(args):
    """Configures and runs the Grounded SAM detector."""
    print("Initializing Grounded SAM detector...")
    model_id = {
        'dino': constants.DINO_MODELS[args.dino_model],
        'sam': constants.SAM_MODELS[args.sam_model]
    }
    class_map = {label: idx for idx, label in enumerate(args.class_names)}
    detector = GroundedSamDetector(model_id)

    if args.precomputed_boxes:
        print(f"Running SAM only, using existing labels from: {args.precomputed_boxes}")
        
        # TODO da rivedere completamente
        return detector.process_precomputed_boxes(
            directory_path=args.bbox_gt,
            model_name=args.model_type,
            class_map=class_map,
            batch_size=args.batch_size
        )
    
    print("Running full Grounded SAM pipeline (DINO + SAM)...")

    return detector.process_directory(
        directory_path=args.seg_gt,
        model_name=args.model_type,
        class_map=class_map,
        batch_size=args.batch_size,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        pred_only=args.save_predictions_only,
        metrics_only=args.save_metrics_only,
        image_size=args.image_size,
        map_thresh_list=args.map_thresh,
    )


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

    subparsers = parser.add_subparsers(dest='model_type', required=True, help='Select the model to run.')

    # --- Grounding DINO Parser ---
    dino_parser = subparsers.add_parser('dino', help='Use Grounding DINO model.', parents=[parent_parser])
    dino_parser.add_argument('--model', type=str, default='GDINO-BASE', choices=constants.DINO_MODELS.keys())
    dino_parser.add_argument('--text-threshold', type=float, default=0.25, help='Confidence threshold for text matching.')
    dino_parser.add_argument('--box-threshold', '-t', type=float, default=0.2, help='Confidence threshold for object detection box.')
    dino_parser.set_defaults(func=run_dino)

    # --- OwlViT Parser ---
    owl_parser = subparsers.add_parser('owl', help='Use OwlViT model.', parents=[parent_parser])
    owl_parser.add_argument('--model', type=str, default='OWLVIT-BASE-32', choices=constants.OWL_MODELS.keys())
    owl_parser.add_argument('--score-threshold', '-t', type=float, default=0.1, help='Detection confidence threshold.')
    owl_parser.add_argument('--prompts', '-p', nargs='+', required=True, help='Descriptive text prompts for detection.')
    owl_parser.set_defaults(func=run_owl)

    # --- YOLO-World Parser ---
    yolo_parser = subparsers.add_parser('yolo_world', help='Use Ultralytics YOLO-World model.', parents=[parent_parser])
    yolo_parser.add_argument('--model', type=str, default='YOLO-X-WORLD', choices=constants.YOLO_WORLD_MODELS.keys())
    yolo_parser.add_argument('--score-threshold', '-t', type=float, default=0.05, help='Detection confidence threshold.')
    yolo_parser.add_argument('--prompts', '-p', nargs='+', required=True, help='Descriptive text prompts for detection.')
    yolo_parser.set_defaults(func=run_yolo_world)

    # --- Grounded SAM Parser ---
    grounded_sam_parser = subparsers.add_parser('grounded_sam', help='Use Grounded SAM pipeline.', parents=[parent_parser])
    grounded_sam_parser.add_argument('--dino-model', type=str, default='GDINO-BASE', choices=constants.DINO_MODELS.keys())
    grounded_sam_parser.add_argument('--sam-model', type=str, default='SAM-2.1', choices=constants.SAM_MODELS.keys())
    grounded_sam_parser.add_argument('--text-threshold', type=float, default=0.25, help='Confidence threshold for text matching.')
    grounded_sam_parser.add_argument('--box-threshold', '-t', type=float, default=0.2, help='Confidence threshold for object detection box.')
    grounded_sam_parser.add_argument(
        '--precomputed-boxes', action="store_true", default=None,
        help='Wether to skip the DINO step and use the boxes in dataset_path/bbox_ground_tr for SAM.'
    )
    grounded_sam_parser.set_defaults(func=run_grounded_sam)

    args = parser.parse_args()

    config = parse_cfg(args.config_path)
    args.__dict__.update(config)
    
    output_root = args.func(args)

    if output_root:
        classes_file_path = Path(output_root) / 'classes.txt'
        with open(classes_file_path, 'w') as f:
            for class_name in args.class_names:
                f.write(class_name.strip() + '\n')
        print(f"Saved class name to {classes_file_path}")
    else:
        print("\nProcessing finished without saving.")
