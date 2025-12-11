import os
import base64
import json
import time
from typing import Optional, Dict, Any
from PIL import Image
from io import BytesIO

from google import genai
from google.genai.errors import ServerError
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from utils import safe_json_extract
from dotenv import load_dotenv

load_dotenv()


class ContextAnalysisInput(BaseModel):
    """Input schema for context analysis tool"""
    image_path: str = Field(..., description="Path to the image file to analyze")
    entity: str = Field(..., description="The entity to be placed in the image (e.g., 'dog', 'cat')")
    context_number: int = Field(..., description="Maximum number of placement scenarios to generate")


class ImageGenerationInput(BaseModel):
    """Input schema for image generation tool"""
    image_path: str = Field(..., description="Path to the base image file")
    entity: str = Field(..., description="The entity to insert into the image")
    context_option: str = Field(..., description="Description of where/how to place the entity")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts for API calls")


class ImageJudgmentInput(BaseModel):
    """Input schema for image judgment tool"""
    image_data: bytes = Field(..., description="PIL Image serialized as bytes")
    entity: str = Field(..., description="The entity that was inserted into the image")


class AnalyzeContextTool(BaseTool):
    name: str = "Analyze Image Context"
    description: str = (
        "Analyzes an image and identifies optimal placement scenarios for inserting a specific entity. "
        "Returns a JSON object with numbered placement options."
    )
    args_schema: type[BaseModel] = ContextAnalysisInput

    def _run(self, image_path: str, entity: str, context_number: int) -> Dict[str, str]:
        """Execute context analysis using Gemini vision model"""
        api_key = os.getenv("API_KEY")
        ai = genai.Client(api_key=api_key)

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


class GenerateImageWithEntityTool(BaseTool):
    name: str = "Generate Image With Entity"
    description: str = (
        "Inserts a specified entity into an image according to a context description. "
        "Uses Gemini image generation model with retry logic for robustness. "
        "Returns the path to the generated image file or None if generation fails."
    )
    args_schema: type[BaseModel] = ImageGenerationInput

    def _run(
        self,
        image_path: str,
        entity: str,
        context_option: str,
        max_retries: int = 3
    ) -> Optional[str]:
        """Execute entity insertion using Gemini image generation model"""
        api_key = os.getenv("API_KEY")
        ai = genai.Client(api_key=api_key)

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
                        # Save to temporary file and return path
                        temp_dir = "/tmp/syngen_crew"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f"generated_{int(time.time())}.png")

                        img = Image.open(BytesIO(part.inline_data.data))
                        img.save(temp_path)
                        return temp_path

            except ServerError as e:
                print(f"ServerError attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(3)
                else:
                    return None
        return None


class JudgeImageQualityTool(BaseTool):
    name: str = "Judge Image Quality"
    description: str = (
        "Evaluates whether a generated entity looks natural and well-integrated into the image. "
        "Uses Gemini vision model as a quality judge. "
        "Returns a JSON object with 'status' boolean indicating approval/rejection."
    )
    args_schema: type[BaseModel] = ImageJudgmentInput

    def _run(self, image_data: bytes, entity: str) -> Dict[str, bool]:
        """Execute quality judgment using Gemini vision model"""
        api_key = os.getenv("API_KEY")
        ai = genai.Client(api_key=api_key)

        base64_image = base64.b64encode(image_data).decode("utf-8")

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


# Tool instances for CrewAI
analyze_context_tool = AnalyzeContextTool()
generate_image_tool = GenerateImageWithEntityTool()
judge_image_tool = JudgeImageQualityTool()
