import os
import base64
import argparse
import json
import shutil
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import albumentations as A
import numpy as np

load_dotenv()

parser = argparse.ArgumentParser(description="Synthetic Image Generation with AI and Data Augmentation + Judge")
parser.add_argument("-e", "--entity", type=str, help="Entity to add in the images", required=True)
parser.add_argument("-c", "--context_limit", type=int, help="The limit for generate contexts", default=3)
parser.add_argument("-i", "--input_folder", type=str, help="Input folder with images", default="input_images")
parser.add_argument("-o", "--output_folder", type=str, help="Output folder for generated images", default="output_images")
parser.add_argument("-d", "--discard_folder", type=str, help="Folder for discarded images", default="discarded_images")
args = parser.parse_args()

API_KEY = os.getenv("API_KEY")
ENTITY = args.entity
CONTEXT_LIMIT = args.context_limit
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
DISCARD_FOLDER = os.path.join(args.discard_folder, ENTITY)
ENTITY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, ENTITY)

os.makedirs(ENTITY_OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(DISCARD_FOLDER):
    shutil.rmtree(DISCARD_FOLDER)
os.makedirs(DISCARD_FOLDER, exist_ok=True)

# ---------------- Data Augmentation ----------------
transform = A.Compose([
    A.HorizontalFlip(p=1)
])

# ---------------- GenAI Client ----------------
ai = genai.Client(api_key=API_KEY)

# ---------------- Functions ----------------
def safe_json_extract(text, entity):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"1": f"{entity} in the scene (fallback)"}

def analyze_context(image_path, entity, context_number):
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt_text = (
        f"Analyze this image and return possible scenarios where the entity '{entity}' could be placed. "
        f"The output must be ONLY a valid JSON object with keys as integers and values as short English descriptions. "
        f"Example: {{\"1\": \"{entity} standing in the roadside\", \"2\": \"{entity} standing in the middle of the road\"}}. "
        f"Limit yourself to a maximum of {context_number} values. Only valid JSON."
    )

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}}
    ]

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text_out = response.candidates[0].content.parts[0].text
    return safe_json_extract(text_out, entity)

def generate_with_entity(image_path, entity, context_option=None, max_retries=3):
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    if context_option:
        prompt_text = f"Add {entity} in this context: {context_option}."
    else:
        prompt_text = f"Add {entity} into the scene."

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = ai.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=prompt
            )
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return Image.open(BytesIO(part.inline_data.data))

        except ServerError as e:
            print(f"ServerError attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                return None
    return None

def judge_image(pil_image, entity):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = (
        f"You are a strict evaluator of AI-generated content. "
        f"Look ONLY at the entity '{entity}' in the image. "
        f"If the entity looks artificial, fake, poorly blended, distorted, or clearly AI-generated, "
        f"respond with this exact JSON: {{\"status\": false}}. "
        f"If the entity looks natural enough in the context of the scene (even if not perfect), "
        f"respond with this exact JSON: {{\"status\": true}}. "
        f"Do not include explanations, only the JSON."
    )

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": "image/png", "data": base64_image}}
    ]

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text_out = response.candidates[0].content.parts[0].text.strip()
    try:
        return json.loads(text_out)
    except Exception:
        return {"status": False}

def augment_image(pil_image):
    img = np.array(pil_image)
    augmented = transform(image=img)["image"]
    return Image.fromarray(augmented)

# ---------------- Main ----------------
start_time = time.time()

report = {
    "entity": ENTITY,
    "total_images": 0,
    "api_success": 0,
    "api_failures": 0,
    "augmented_images": 0,
    "discarded": 0,
    "contexts": {}
}

for img_file in tqdm(os.listdir(INPUT_FOLDER)):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        report["total_images"] += 1
        input_path = os.path.join(INPUT_FOLDER, img_file)

        contexts = analyze_context(input_path, ENTITY, CONTEXT_LIMIT)
        report["contexts"][img_file] = contexts

        for idx, context_option in contexts.items():
            output_image = generate_with_entity(input_path, ENTITY, context_option)
            if not output_image:
                report["api_failures"] += 1
                continue

            report["api_success"] += 1

            judge_result = judge_image(output_image, ENTITY)
            if not judge_result.get("status", False):
                discard_name = f"{os.path.splitext(img_file)[0]}_ctx{idx}.png"
                discard_path = os.path.join(DISCARD_FOLDER, discard_name)
                output_image.save(discard_path)
                report["discarded"] += 1
                continue

            base_name, ext = os.path.splitext(img_file)
            output_filename = f"{base_name}_ctx{idx}{ext}"
            output_path = os.path.join(ENTITY_OUTPUT_FOLDER, output_filename)
            output_image.save(output_path)

            aug_image = augment_image(output_image)
            aug_filename = f"{base_name}_ctx{idx}_aug{ext}"
            aug_path = os.path.join(ENTITY_OUTPUT_FOLDER, aug_filename)
            aug_image.save(aug_path)
            report["augmented_images"] += 1

elapsed = time.time() - start_time
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
report["processing_time"] = f"{int(h)}h {int(m)}m {int(s)}s"

report_path = os.path.join(ENTITY_OUTPUT_FOLDER, "report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Report saved in {report_path}")
print(f"Total processing time: {report['processing_time']}")
