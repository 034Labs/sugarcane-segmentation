import cv2
import time

from .inference import process_frame


def run_video(args, model, target_id, mask_annotator, label_annotator):
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Erro ao abrir video: {args.video}")

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

    print(f"Processando video: {args.video}")
    print(f"Saida: {output_path}")

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
            print(f"[{frame_idx} frames] Deteccoes acumuladas: {total_detections}")

    cap.release()
    out.release()

    if frame_idx == 0:
        print("\n=== FINALIZADO ===")
        print("Nenhum frame foi processado. Verifique se o video esta vazio ou corrompido.")
        print(f"Arquivo de saida criado em: {output_path}")
        return

    end_total = time.perf_counter()

    total_time = end_total - start_total
    avg_inf_time = total_inf_time / frame_idx
    inf_fps = 1.0 / avg_inf_time
    pipeline_fps = frame_idx / total_time

    print("\n=== PERFORMANCE ===")
    print(f"Inferencia media: {avg_inf_time*1000:.2f} ms")
    print(f"FPS de inferencia: {inf_fps:.2f}")
    print(f"Tempo total pipeline: {total_time:.2f} s")
    print(f"FPS total (pipeline): {pipeline_fps:.2f}")
    print(f"% tempo em inferencia: {(total_inf_time / total_time)*100:.1f}%")

    print("\n=== FINALIZADO ===")
    print(f"Frames processados: {frame_idx}")
    print(f"Total detections (frame-wise): {total_detections}")


def run_image(args, model, target_id, mask_annotator, label_annotator):
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
    print(f"{count} instancias de '{args.target_class}' encontradas")
    print(f"Tempo de inferencia: {inf_time*1000:.2f} ms")
    print(f"Salvo em: {output_path}")
