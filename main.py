import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()
model.optimize_for_inference()
detections = model.predict("images/junior-galdino-066.jpg", threshold=0.5)

person_class_id = [k for k, v in COCO_CLASSES.items() if v == "person"][0]
person_count = sum(1 for class_id in detections.class_id if class_id == person_class_id)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Add person count with background using Pillow
text = f"Person: {person_count}"
pil_image = Image.fromarray(annotated_image)
draw = ImageDraw.Draw(pil_image)
font = ImageFont.truetype("arial.ttf", 24)

text_bbox = draw.textbbox((0, 0), text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]

x, y = 10, 10
draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0))
draw.text((x, y), text, font=font, fill=(0, 255, 0))

pil_image.save("results/resultado.jpg")
print(f"Imagem salva com sucesso como resultado.jpg! Pessoas detectadas: {person_count}")
