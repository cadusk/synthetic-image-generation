import os
import base64
from dotenv import load_dotenv
from google import genai
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np

# ---------------- Carrega variáveis do .env ----------------
load_dotenv()

API_KEY = os.getenv("API_KEY")
NUM_OBJECTS_TO_ADD = int(os.getenv("NUM_OBJECTS_TO_ADD", 1))
INPUT_FOLDER = os.getenv("INPUT_FOLDER", "input_images")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output_images")
AUGMENTED_FOLDER = os.getenv("AUGMENTED_FOLDER", "augmented_images")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# ---------------- Data Augmentation ----------------
transform = A.Compose([
    A.HorizontalFlip(p=1)
])

# ---------------- Cliente GenAI ----------------
ai = genai.Client(api_key=API_KEY)

# ---------------- Funções ----------------
def generate_with_animals(image_path, num_objects=NUM_OBJECTS_TO_ADD):
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    # usar IA para fazer prompts dinamicos
    # separar entidades, animais, fogos, etc
    # contextualizar atividades das entidades
    
    prompt_text = (
        f"Add {num_objects} realistic animals (dogs or cows) to the highway in this image, "
        "carefully considering the perspective, lighting, shadows, and proportions relative "
        "to the existing vehicles and road. Maintain natural colors, realistic sizes, "
        "and proper integration into the environment, keeping motion blur for moving objects, "
        "reflections on wet surfaces if any, and correct lighting effects."
    )

    prompt = [
        {"text": prompt_text},
        {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64_image
            }
        }
    ]

    response = ai.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=prompt
    )

    parts = response.candidates[0].content.parts
    for part in parts:
        if hasattr(part, "inline_data") and part.inline_data:
            image_bytes = part.inline_data.data  # já é bytes
            return Image.open(BytesIO(image_bytes))
    return None

def augment_image(pil_image):
    img = np.array(pil_image)
    augmented = transform(image=img)["image"]
    return Image.fromarray(augmented)

# ---------------- Execução Principal ----------------
report = {
    "total_images": 0,
    "api_success": 0,
    "api_failures": 0,
    "augmented_images": 0
}

for img_file in tqdm(os.listdir(INPUT_FOLDER)):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        report["total_images"] += 1
        input_path = os.path.join(INPUT_FOLDER, img_file)
        
        output_image = generate_with_animals(input_path)
        if output_image:
            report["api_success"] += 1
            output_path = os.path.join(OUTPUT_FOLDER, img_file)
            output_image.save(output_path)
            
            aug_image = augment_image(output_image)
            aug_path = os.path.join(AUGMENTED_FOLDER, f"aug_{img_file}")
            aug_image.save(aug_path)
            report["augmented_images"] += 1
        else:
            report["api_failures"] += 1

# ---------------- Relatório ----------------
report_path = "report.txt"
with open(report_path, "w") as f:
    f.write("=== Relatorio de Imagens ===\n")
    for key, value in report.items():
        f.write(f"{key}: {value}\n")

print(f"Relatório gerado em {report_path}")
