from crewai import Task
from typing import Dict, Any


def create_context_analysis_task(
    agent,
    image_path: str,
    entity: str,
    context_limit: int
) -> Task:
    """
    Create a task for analyzing image context and identifying placement scenarios.
    """
    return Task(
        description=(
            f"Analyze the image at '{image_path}' to identify optimal placement scenarios "
            f"for inserting a '{entity}'. Generate up to {context_limit} different context "
            f"options that describe where and how the entity could be naturally placed. "
            f"Consider factors like perspective, scale, lighting, and environmental appropriateness. "
            f"Output must be valid JSON with numbered keys (e.g., {{'1': 'description', '2': 'description'}})."
        ),
        agent=agent,
        expected_output=(
            f"A JSON object with {context_limit} or fewer placement scenarios, each describing "
            f"a specific context for inserting the {entity} into the scene."
        )
    )


def create_image_generation_task(
    agent,
    image_path: str,
    entity: str,
    context_description: str,
    context_idx: str
) -> Task:
    """
    Create a task for generating a synthetic image with an inserted entity.
    """
    return Task(
        description=(
            f"Generate a synthetic image by inserting a '{entity}' into the base image at '{image_path}'. "
            f"Follow this specific placement context: '{context_description}'. "
            f"Ensure the entity is properly scaled relative to other objects in the scene. "
            f"Maintain consistent lighting, shadows, and perspective. "
            f"Do not modify other objects in the original scene. "
            f"Use retry logic (up to 3 attempts) if API errors occur. "
            f"This is context option {context_idx}."
        ),
        agent=agent,
        expected_output=(
            f"Path to the generated image file containing the {entity} inserted according to the context, "
            f"or None if generation fails after retries."
        )
    )


def create_quality_review_task(
    agent,
    generated_image_path: str,
    entity: str,
    base_filename: str,
    context_idx: str,
    output_folder: str,
    discard_folder: str
) -> Task:
    """
    Create a task for reviewing the quality of a generated image.
    """
    return Task(
        description=(
            f"Evaluate the quality and realism of the generated image at '{generated_image_path}'. "
            f"Focus specifically on the '{entity}' that was inserted into the scene. "
            f"Check for:\n"
            f"1. Proper scale and proportionality relative to other objects\n"
            f"2. Natural blending with the background (no harsh edges or artifacts)\n"
            f"3. Consistent lighting and shadows\n"
            f"4. Absence of obvious AI-generation markers (unnatural smoothness, distortions)\n"
            f"5. Overall photorealistic appearance\n\n"
            f"If the image passes quality standards, it should be saved to: {output_folder}\n"
            f"If the image fails quality standards, it should be saved to: {discard_folder}\n"
            f"Use filename: {base_filename}_ctx{context_idx}.png for discarded images\n"
            f"Provide a clear approval/rejection decision with reasoning."
        ),
        agent=agent,
        expected_output=(
            f"A JSON object with 'status' (true/false) indicating approval/rejection, "
            f"and the final path where the image was saved (output or discard folder)."
        )
    )


def create_data_augmentation_task(
    agent,
    approved_image_path: str,
    entity: str,
    base_filename: str,
    context_idx: str,
    output_folder: str,
    should_augment: bool
) -> Task:
    """
    Create a task for augmenting approved images.
    """
    return Task(
        description=(
            f"Process the approved image at '{approved_image_path}' for dataset enhancement. "
            f"Save the original to the output folder with proper naming. "
            f"{'Then apply data augmentation (horizontal flip) and save the augmented version with _aug suffix.' if should_augment else 'Skip augmentation as it is disabled.'} "
            f"Use filename pattern: {base_filename}_ctx{context_idx}.ext for original, "
            f"{base_filename}_ctx{context_idx}_aug.ext for augmented. "
            f"Ensure all files are saved to: {output_folder}"
        ),
        agent=agent,
        expected_output=(
            f"Confirmation of saved files with their full paths. "
            f"Include both original and augmented paths if augmentation was applied."
        )
    )


def create_folder_setup_task(
    agent,
    output_folder: str,
    discard_folder: str
) -> Task:
    """
    Create a task for setting up output and discard folders.
    """
    return Task(
        description=(
            f"Set up the directory structure for the synthetic image generation pipeline. "
            f"Create the output folder at: {output_folder} "
            f"Create/clear the discard folder at: {discard_folder} "
            f"Ensure both folders are ready to receive generated images."
        ),
        agent=agent,
        expected_output=(
            f"Confirmation that folders have been created and are ready for use. "
            f"Include the full paths of both folders."
        )
    )


def create_report_generation_task(
    agent,
    report_data: Dict[str, Any],
    output_folder: str
) -> Task:
    """
    Create a task for generating and saving the pipeline report.
    """
    return Task(
        description=(
            f"Generate a comprehensive report summarizing the synthetic image generation pipeline results. "
            f"Include the following metrics from the provided data: "
            f"- Entity: {report_data.get('entity', 'N/A')} "
            f"- Total images processed: {report_data.get('total_images', 0)} "
            f"- API successes: {report_data.get('api_success', 0)} "
            f"- API failures: {report_data.get('api_failures', 0)} "
            f"- Augmented images: {report_data.get('augmented_images', 0)} "
            f"- Discarded images: {report_data.get('discarded', 0)} "
            f"- Processing time: {report_data.get('processing_time', 'N/A')} "
            f"- Context descriptions per image "
            f"Save the report as JSON to: {output_folder}/report.json "
            f"Ensure the report is properly formatted and human-readable."
        ),
        agent=agent,
        expected_output=(
            f"Path to the saved report.json file with confirmation of successful write."
        )
    )
