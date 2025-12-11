import os
import numpy as np
from typing import Optional
from PIL import Image
import albumentations as A

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ImageAugmentationInput(BaseModel):
    """Input schema for image augmentation tool"""
    image_path: str = Field(..., description="Path to the image file to augment")
    output_path: str = Field(..., description="Path where the augmented image should be saved")


class ImageSaveInput(BaseModel):
    """Input schema for image save tool"""
    source_path: str = Field(..., description="Path to the source image file")
    destination_path: str = Field(..., description="Path where the image should be saved")


class ImageLoadInput(BaseModel):
    """Input schema for image load tool"""
    image_path: str = Field(..., description="Path to the image file to load")


class AugmentImageTool(BaseTool):
    name: str = "Augment Image"
    description: str = (
        "Applies data augmentation transformations to an image for ML training dataset enhancement. "
        "Currently applies horizontal flip. Returns the path to the augmented image."
    )
    args_schema: type[BaseModel] = ImageAugmentationInput

    def _run(self, image_path: str, output_path: str) -> str:
        """Apply data augmentation to an image"""
        # Define transformation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=1)
        ])

        # Load image
        pil_image = Image.open(image_path)
        img = np.array(pil_image)

        # Apply transformation
        augmented = transform(image=img)["image"]
        aug_image = Image.fromarray(augmented)

        # Save augmented image
        aug_image.save(output_path)
        return output_path


class SaveImageTool(BaseTool):
    name: str = "Save Image"
    description: str = (
        "Saves an image from one location to another, preserving format. "
        "Useful for organizing approved/rejected images into appropriate folders."
    )
    args_schema: type[BaseModel] = ImageSaveInput

    def _run(self, source_path: str, destination_path: str) -> str:
        """Save/copy an image to a destination path"""
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Load and save image
        image = Image.open(source_path)
        image.save(destination_path)
        return destination_path


class LoadImageTool(BaseTool):
    name: str = "Load Image"
    description: str = (
        "Loads an image from a file path and returns basic metadata. "
        "Useful for verifying image properties before processing."
    )
    args_schema: type[BaseModel] = ImageLoadInput

    def _run(self, image_path: str) -> dict:
        """Load an image and return its metadata"""
        image = Image.open(image_path)
        return {
            "path": image_path,
            "size": image.size,
            "mode": image.mode,
            "format": image.format
        }


# Tool instances for CrewAI
augment_image_tool = AugmentImageTool()
save_image_tool = SaveImageTool()
load_image_tool = LoadImageTool()
