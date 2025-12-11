import os
import json
import shutil
from typing import Dict, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FolderSetupInput(BaseModel):
    """Input schema for folder setup tool"""
    output_folder: str = Field(..., description="Path to the output folder")
    discard_folder: str = Field(..., description="Path to the discard folder")


class ReportSaveInput(BaseModel):
    """Input schema for report save tool"""
    report_data: Dict[str, Any] = Field(..., description="Report data as dictionary")
    output_folder: str = Field(..., description="Folder where report.json should be saved")


class ListImagesInput(BaseModel):
    """Input schema for list images tool"""
    folder_path: str = Field(..., description="Path to the folder containing images")


class SetupFoldersTool(BaseTool):
    name: str = "Setup Folders"
    description: str = (
        "Creates output and discard folders for organizing generated images. "
        "Clears the discard folder if it exists to ensure a fresh start."
    )
    args_schema: type[BaseModel] = FolderSetupInput

    def _run(self, output_folder: str, discard_folder: str) -> str:
        """Setup output and discard folders"""
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Clear and recreate discard folder
        if os.path.exists(discard_folder):
            shutil.rmtree(discard_folder)
        os.makedirs(discard_folder, exist_ok=True)

        return f"Folders created: output={output_folder}, discard={discard_folder}"


class SaveReportTool(BaseTool):
    name: str = "Save Report"
    description: str = (
        "Saves a processing report as JSON file in the output folder. "
        "The report includes metrics like total images, successes, failures, and processing time."
    )
    args_schema: type[BaseModel] = ReportSaveInput

    def _run(self, report_data: Dict[str, Any], output_folder: str) -> str:
        """Save report to JSON file"""
        report_path = os.path.join(output_folder, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        return report_path


class ListImagesTool(BaseTool):
    name: str = "List Images"
    description: str = (
        "Lists all image files (PNG, JPG, JPEG) in a specified folder. "
        "Returns a list of filenames for batch processing."
    )
    args_schema: type[BaseModel] = ListImagesInput

    def _run(self, folder_path: str) -> list:
        """List all image files in a folder"""
        if not os.path.exists(folder_path):
            return []

        image_extensions = ('.png', '.jpg', '.jpeg')
        images = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
                images.append(filename)

        return images


# Tool instances for CrewAI
setup_folders_tool = SetupFoldersTool()
save_report_tool = SaveReportTool()
list_images_tool = ListImagesTool()
