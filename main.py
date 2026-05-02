import supervision as sv
import cv2
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()

detections = model.predict("junior-galdino-066.jpg", threshold=0.5)

person_class_id = [k for k, v in COCO_CLASSES.items() if v == "person"][0]
person_count = sum(1 for class_id in detections.class_id if class_id == person_class_id)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Add person count with background
text = f"Person: {person_count}"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
x, y = 10, 30
cv2.rectangle(annotated_image, (x-5, y-text_height-5), (x+text_width+5, y+baseline+5), (0, 0, 0), -1)
cv2.putText(annotated_image, text, (x, y), font, font_scale, (0, 255, 0), thickness)

cv2.imwrite("resultado.jpg", annotated_image)
print(f"Imagem salva com sucesso como resultado.jpg! Pessoas detectadas: {person_count}")
