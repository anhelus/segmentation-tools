# Segmentation Tools

A comprehensive research toolkit for zero-shot object detection and segmentation, designed to facilitate the evaluation, optimization, and fine-tuning of leading vision foundation models.

## Overview

This repository provides a unified interface for working with state-of-the-art zero-shot models. It is built to support research workflows, allowing for:

*   **Zero-Shot Inference**: Seamlessly switch between different detection and segmentation backbones.
*   **Fine-tuning**: Tools for adapting models (e.g., YOLO-World) to custom datasets.
*   **Metric Evaluation**: Robust calculation of standard detection and segmentation metrics.
*   **Hyperparameter Optimization**: Automated threshold tuning to maximize custom metrics like mAP.

## Supported Models

The toolkit supports a variety of models for open-vocabulary detection and segmentation.

| Family | Inference | Training | Models | Description |
| :---   | :---      | :---     | :---   |  :---       |
| **Grounding DINO** | ✅ | ❌ | `GDINO-TINY`, `GDINO-BASE` | Open-set object detection using text prompts. |
| **Grounded SAM**   | ✅ | ❌ | `SAM-2-B`, `SAM-2.1-L`, etc. | Combines Grounding DINO with Segment Anything Model 2/2.1 for zero-shot instance segmentation. |
| **OWL / OWLv2**    | ✅ | ❌ | `OWLVIT-BASE`, `OWL2-LARGE` | Google's Open-World Localization ViT models. |
| **YOLO-World**     | ✅ | ✅ | `YOLO-v8[s/m/l/x]-world` | Real-time open-vocabulary detection from Ultralytics. |

> For further information, see [here](src/models/constants.py "Model dictionary").

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/anhelus/segmentation-tools.git
    cd segmentation-tools
    ```
2.  [Optional] Create and activate the virtual environment
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install strict dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a CUDA-compatible PyTorch installation.*

4.  Download necessary model weights into the `checkpoints/` directory as required by the specific models you intend to use (e.g., SAM weights, YOLO-World weights).

## Project Structure

```
├── checkpoints/        # Model weights storage
├── src/
│   ├── metrics/        # Metric calculation strategies
│   └── models/         # Model implementations (GDINO, SAM, YOLO, OWL)
├── README.md
├── requirements.txt
├── main.py             # Primary inference & evaluation script
├── optim.py            # Hyperparameter optimization script
├── train.py            # Model fine-tuning script
└── *.sh                # Example usage scripts
```

## Usage

### 1. Inference & Evaluation (`main.py`)

Run zero-shot inference on a dataset defined in a YAML config file. The script supports various subcommands for different model families.

**Grounding DINO**
```bash
python3 main.py dino data_config.yaml --model GDINO-BASE --save-metrics-only
```

**Grounded SAM (Detection + Segmentation)**
```bash
python3 main.py grounded_sam data_config.yaml \
    --dino-model GDINO-TINY \
    --sam-model SAM-B
```

**YOLO-World**
```bash
python3 main.py yolo_world data_config.yaml \
    --model YOLO-L-WORLD \
    --prompts "class_1" "class_2"
```

### 2. Hyperparameter Optimization (`optim.py`)

Automatically search for the optimal confidence and box thresholds to maximize performance on your validation set.

```bash
python3 optim.py dino data_config.yaml --output_name optimization_results
```
*This module iterates through parameter grids to find the configuration that yields the highest mAP.*

### 3. Training (`train.py`)

Fine-tune models on your specific dataset. See [above](#supported-models) for model support.

```bash
python3 train.py yolo_world config.yaml \
    --yaml-path dataset.yaml \
    --model YOLO-S-WORLD \
    --prompts "custom_class"
```

## Configuration

Inference and optimization scripts rely on a YAML configuration file to define dataset paths and other runtime parameters. Ensure your data configuration matches the expected format (compatible with standard Ultralytics/COCO formats where applicable).

## Citation

If you use this toolkit in your research, please cite the repository:

```bibtex
@misc{segmentation-tools,
  author = {Anhelus},
  title = {Segmentation Tools: A Zero-Shot Detection & Segmentation Toolkit},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anhelus/segmentation-tools}}
}
```
