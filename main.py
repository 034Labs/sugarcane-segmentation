import supervision as sv
import cv2
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

cv2.imwrite("resultado.jpg", annotated_image)
print("Imagem salva com sucesso como resultado.jpg!")
