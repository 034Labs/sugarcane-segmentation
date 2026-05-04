import supervision as sv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

# 1. Inicialização do Modelo
model = RFDETRSegMedium()
model.optimize_for_inference()

# 2. Inferência de Teste
IMAGE_PATH = "images/junior-galdino-066.jpg"
detections = model.predict(IMAGE_PATH, threshold=0.5)

# 3. Lógica de Filtragem
TARGET_CLASS = "person"
class_id_map = {v: k for k, v in COCO_CLASSES.items()}
target_id = class_id_map.get(TARGET_CLASS)

# Filtrando as detecções apenas para a classe "person"
mask = detections.class_id == target_id
target_count = np.sum(mask)

# 4. Anotação Automática com Supervision
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [f"{TARGET_CLASS} {conf:.2f}" for conf in detections.confidence[mask]]
source_img = detections.metadata["source_image"]

# Aplicando as máscaras e depois os rótulos
annotated_img = mask_annotator.annotate(scene=source_img, detections=detections[mask])
annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections[mask], labels=labels)

# 5. Adição de Texto com Pillow
pil_image = Image.fromarray(annotated_img)
draw = ImageDraw.Draw(pil_image)

# Fallback de fonte para garantir que funcione em qualquer sistema (Windows/Linux/Colab).
try:
    font = ImageFont.truetype("arial.ttf", 26)
except IOError:
    font = ImageFont.load_default()

text = f"Detecções ({TARGET_CLASS}): {target_count}"
draw.rectangle([5, 5, 300, 45], fill=(0, 0, 0))
draw.text((15, 10), text, font=font, fill=(0, 255, 0))

# 6. Salvamento do Resultado
pil_image.save("results/teste_aprendizado.jpg")
print(f"Sucesso! {target_count} instâncias de '{TARGET_CLASS}' encontradas.")