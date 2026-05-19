import argparse
import os

import supervision as sv
import torch

from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

from .pipeline import run_image, run_video


def build_parser():
    parser = argparse.ArgumentParser(
        description='RF-DETR Segmentation com medicao de performance'
    )

    parser.add_argument('--image', type=str, default='images/input.jpg')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--class', dest='target_class', type=str, default='person')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output', type=str, default='results/output')
    parser.add_argument('--resize', type=float, default=1.0)
    return parser


def run():
    parser = build_parser()
    args = parser.parse_args()

    class_id_map = {v: k for k, v in COCO_CLASSES.items()}
    target_id = class_id_map.get(args.target_class)

    if target_id is None:
        raise ValueError(f"Classe '{args.target_class}' nao encontrada no COCO.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")

    model = RFDETRSegMedium(device=device)

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)

    if args.video:
        run_video(args, model, target_id, mask_annotator, label_annotator)
    else:
        run_image(args, model, target_id, mask_annotator, label_annotator)
