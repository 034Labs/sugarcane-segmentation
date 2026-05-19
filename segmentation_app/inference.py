import cv2
import numpy as np
import time
import torch


def process_frame(model, frame_bgr, target_id, threshold, target_class,
                  mask_annotator, label_annotator):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_inf = time.perf_counter()

    with torch.no_grad():
        detections = model.predict(frame_rgb, threshold=threshold)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_inf = time.perf_counter()
    inference_time = end_inf - start_inf

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

    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR), target_count, inference_time
