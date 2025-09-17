import base64
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def main():
    ai = genai.Client(api_key=os.getenv("API_KEY"))

    input_image_path = "cat.jpg"
    with open(input_image_path, "rb") as f:
        image_data = f.read()
    
    ext = os.path.splitext(input_image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = [
        {
            "text": (
                "Using the image of the cat, create a photorealistic, "
                "street-level view of the cat walking along a sidewalk in a "
                "New York City neighborhood, with the blurred legs of pedestrians "
                "and yellow cabs passing by in the background."
            )
        },
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
    print("Número de partes recebidas:", len(parts))

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for i, part in enumerate(parts):
        # Ajuste para a nova estrutura
        if hasattr(part, "text") and part.text:
            print(f"[Part {i} - texto]: {part.text}")
        elif hasattr(part, "inline_data") and part.inline_data:
            print(f"[Part {i} - imagem recebida] Salvando...")
            image_bytes = part.inline_data.data  # já é bytes
            output_path = os.path.join(output_folder, f"cat_generated_{i}.png")
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Imagem salva em: {output_path}")
        else:
            print(f"[Part {i}] Conteúdo desconhecido:", part)

if __name__ == "__main__":
    main()
