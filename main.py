import argparse
import os
import cv2
import numpy as np
import supervision as sv
import torch
import time

from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


def process_frame(model, frame_bgr, target_id, threshold, target_class,
                  mask_annotator, label_annotator):

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ===== INFERÊNCIA =====
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_inf = time.perf_counter()

    with torch.no_grad():
        detections = model.predict(frame_rgb, threshold=threshold)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_inf = time.perf_counter()
    inference_time = end_inf - start_inf

    # ===== FILTRO =====
    mask = detections.class_id == target_id
    target_count = int(np.sum(mask))

    if target_count > 0:
        det_filtered = detections[mask]

        labels = [
            f"{target_class} {conf:.2f}"
            for conf in det_filtered.confidence
        ]

        annotated_rgb = mask_annotator.annotate(
            scene=frame_rgb,
            detections=det_filtered
        )

        annotated_rgb = label_annotator.annotate(
            scene=annotated_rgb,
            detections=det_filtered,
            labels=labels
        )
    else:
        annotated_rgb = frame_rgb

    # ===== OVERLAY =====
    text = f"Deteccoes ({target_class}): {target_count}"
    cv2.rectangle(annotated_rgb, (5, 5), (320, 45), (0, 0, 0), -1)
    cv2.putText(
        annotated_rgb,
        text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # RGB -> BGR
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR), target_count, inference_time


def main():
    parser = argparse.ArgumentParser(
        description='RF-DETR Segmentation com medição de performance'
    )

    parser.add_argument('--image', type=str, default='images/input.jpg')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--class', dest='target_class', type=str, default='person')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output', type=str, default='results/output')
    parser.add_argument('--resize', type=float, default=1.0)

    args = parser.parse_args()

    # ===== CLASSE =====
    class_id_map = {v: k for k, v in COCO_CLASSES.items()}
    target_id = class_id_map.get(args.target_class)

    if target_id is None:
        raise ValueError(f"Classe '{args.target_class}' não encontrada no COCO.")

    # ===== DEVICE =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    # ===== MODELO =====
    model = RFDETRSegMedium(device=device)
    #model.optimize_for_inference()

    # ===== ANNOTATORS =====
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    # ===== OUTPUT DIR =====
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # ===== VÍDEO ============
    # =========================
    if args.video:
        cap = cv2.VideoCapture(args.video)

        if not cap.isOpened():
            raise RuntimeError(f"Erro ao abrir vídeo: {args.video}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        out_width = int(width * args.resize)
        out_height = int(height * args.resize)

        output_path = args.output + '.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        frame_idx = 0
        total_detections = 0
        total_inf_time = 0

        print(f"Processando vídeo: {args.video}")
        print(f"Saída: {output_path}")

        # ===== INÍCIO MEDIÇÃO TOTAL =====
        start_total = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.resize != 1.0:
                frame = cv2.resize(frame, (out_width, out_height))

            annotated_frame, count, inf_time = process_frame(
                model,
                frame,
                target_id,
                args.threshold,
                args.target_class,
                mask_annotator,
                label_annotator
            )

            out.write(annotated_frame)

            total_detections += count
            total_inf_time += inf_time
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"[{frame_idx} frames] Detecções acumuladas: {total_detections}")

        cap.release()
        out.release()

        # ===== FIM MEDIÇÃO TOTAL =====
        end_total = time.perf_counter()

        total_time = end_total - start_total
        avg_inf_time = total_inf_time / frame_idx
        inf_fps = 1.0 / avg_inf_time
        pipeline_fps = frame_idx / total_time

        # ===== RESULTADOS =====
        print("\n=== PERFORMANCE ===")
        print(f"Inferência média: {avg_inf_time*1000:.2f} ms")
        print(f"FPS de inferência: {inf_fps:.2f}")
        print(f"Tempo total pipeline: {total_time:.2f} s")
        print(f"FPS total (pipeline): {pipeline_fps:.2f}")
        print(f"% tempo em inferência: {(total_inf_time / total_time)*100:.1f}%")

        print("\n=== FINALIZADO ===")
        print(f"Frames processados: {frame_idx}")
        print(f"Total detections (frame-wise): {total_detections}")

    # =========================
    # ===== IMAGEM ===========
    # =========================
    else:
        frame = cv2.imread(args.image)

        if frame is None:
            raise RuntimeError(f"Erro ao carregar imagem: {args.image}")

        if args.resize != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * args.resize), int(h * args.resize)))

        annotated_frame, count, inf_time = process_frame(
            model,
            frame,
            target_id,
            args.threshold,
            args.target_class,
            mask_annotator,
            label_annotator
        )

        output_path = args.output + '.jpg'
        cv2.imwrite(output_path, annotated_frame)

        print("\n=== RESULTADO ===")
        print(f"{count} instâncias de '{args.target_class}' encontradas")
        print(f"Tempo de inferência: {inf_time*1000:.2f} ms")
        print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    main()