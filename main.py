import os
import base64
import json
import time
from google import genai
from google.genai.errors import ServerError
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import albumentations as A
import numpy as np

from utils import safe_json_extract
from utils import ensure_folders

from dotenv import load_dotenv
load_dotenv()

from arguments import parse_arguments
args = parse_arguments()

ENTITY = args.entity
CONTEXT_LIMIT = args.context_limit
INPUT_FOLDER = args.input_folder
DISCARD_FOLDER = os.path.join(args.discard_folder, ENTITY)
OUTPUT_FOLDER = os.path.join(args.output_folder, ENTITY)
AUGMENT_IMAGE = args.augment_image


API_KEY = os.getenv("API_KEY")
ai = genai.Client(api_key=API_KEY)


# ---------------- Data Augmentation ----------------
transform = A.Compose([
    A.HorizontalFlip(p=1)
])


# ---------------- Functions ----------------
def analyze_context(image_path, entity, context_number):
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = f"""
        Analyze this image and return possible scenarios where the entity '{entity}' could be placed.
        The output must be ONLY a valid JSON object with keys as integers and values as short English descriptions.
        Example: {{"1": "{entity} standing in the roadside", "2": "{entity} standing in the middle of the road"}}.
        Limit yourself to a maximum of {context_number} values. Only valid JSON.
    """

    contents = [
        {"text": prompt},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}}
    ]

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    text_out = response.candidates[0].content.parts[0].text
    return safe_json_extract(text_out, entity)


def generate_with_entity(image_path, entity, context_option=None, max_retries=3):
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = f"""
        Add {entity} in this context: {context_option}.
        Ensure that the entity's size is proportional to the scene and other objects around it.
        DO NOT make adjustments to other original objects to accommodate the new entity.
    """

    contents = [
        {"text": prompt},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = ai.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=contents
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

    prompt = f"""
        You are a strict evaluator of AI-generated content.
        Look ONLY at the entity '{entity}' in the image.
        If the entity looks artificial, fake, poorly blended, distorted, it's size is not proportial compared to other objects or clearly AI-generated,
        respond with this exact JSON: {{"status": false}}.
        If the entity looks natural enough in the context of the scene (even if not perfect),
        respond with this exact JSON: {{"status": true}}.
        Do not include explanations, only the JSON.
    """

    contents = [
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": base64_image}}
    ]

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    text_out = response.candidates[0].content.parts[0].text.strip()
    try:
        return json.loads(text_out)
    except Exception:
        return {"status": False}



# ---------------- Main ----------------
def main():
    ensure_folders(OUTPUT_FOLDER, DISCARD_FOLDER)

    report = {
        "entity": ENTITY,
        "total_images": 0,
        "api_success": 0,
        "api_failures": 0,
        "augmented_images": 0,
        "discarded": 0,
        "contexts": {}
    }

    start_time = time.time()
    for filename in tqdm(os.listdir(INPUT_FOLDER)):
        
        input_path = os.path.join(INPUT_FOLDER, filename)

        if not os.path.isfile(input_path):
            print("Discarding unsupported dir content: ", input_path)
            continue


        if not input_path.lower().endswith((".png", ".jpg", ".jpeg")):
            print("Discarding unsupported file: ", input_path)
            continue

        report["total_images"] += 1

        contexts = analyze_context(input_path, ENTITY, CONTEXT_LIMIT)
        report["contexts"][filename] = contexts

        for idx, context_option in contexts.items():
            output_image = generate_with_entity(input_path, ENTITY, context_option)
            if not output_image:
                report["api_failures"] += 1
                continue

            report["api_success"] += 1

            judge_result = judge_image(output_image, ENTITY)
            if not judge_result.get("status", False):
                # Judge doesn't like it, send it to discard pile
                discard_name = f"{os.path.splitext(filename)[0]}_ctx{idx}.png"
                discard_path = os.path.join(DISCARD_FOLDER, discard_name)
                output_image.save(discard_path)
                report["discarded"] += 1
                continue

            # send it to good pile
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_ctx{idx}{ext}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            output_image.save(output_path)


            if not AUGMENT_IMAGE:
                continue

            # Let's augment this image applying some transformations
            aug_image = augment_image(output_image)
            aug_filename = f"{base_name}_ctx{idx}_aug{ext}"
            aug_path = os.path.join(OUTPUT_FOLDER, aug_filename)
            aug_image.save(aug_path)
            report["augmented_images"] += 1

    elapsed_time = time.time() - start_time
    report["processing_time"] = format_elapsed_time(elapsed_time)
    save_report(report)



def format_elapsed_time(elapsed_time):
    h, rem = divmod(elapsed_time, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def save_report(report):
    report_path = os.path.join(OUTPUT_FOLDER, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved in {report_path}")


def augment_image(pil_image):
    img = np.array(pil_image)
    augmented = transform(image=img)["image"]
    return Image.fromarray(augmented)



if __name__ == '__main__':
    main()
