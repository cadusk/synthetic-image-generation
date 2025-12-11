import os
from crewai import Agent, LLM
from tools.image_tools import augment_image_tool, save_image_tool
from tools.file_tools import setup_folders_tool


def create_data_engineer_agent() -> Agent:
    """
    Creates a Data Engineer Agent responsible for organizing, augmenting,
    and managing the final dataset of approved synthetic images.
    """
    # Configure Gemini LLM for the agent
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("API_KEY")
    )

    return Agent(
        role='Dataset Enhancement Specialist',
        goal=(
            'Organize approved synthetic images into proper directory structures, apply data '
            'augmentation techniques to expand the training dataset, and ensure all files are '
            'correctly named and formatted for downstream ML training pipelines.'
        ),
        backstory=(
            'You are a seasoned ML data engineer with deep expertise in building and maintaining '
            'high-quality training datasets for computer vision models. You understand that raw data '
            'is just the starting point - proper organization, augmentation, and metadata management '
            'are what transform good data into great datasets. Your experience spans autonomous vehicles, '
            'robotics, and large-scale vision systems where dataset quality directly impacts model safety '
            'and performance. You are meticulous about file naming conventions, directory structures, '
            'and maintaining clear separation between approved outputs and rejected samples. You know '
            'that data augmentation is a powerful technique for improving model robustness, so you apply '
            'transformations like flipping, rotation, and color adjustments strategically. You also '
            'understand the importance of preserving original image quality while creating augmented variants. '
            'Your work ensures that data scientists and ML engineers can easily consume the synthetic dataset '
            'without worrying about file organization or preprocessing. You take pride in delivering '
            'clean, well-structured datasets that are ready for immediate training use.'
        ),
        tools=[augment_image_tool, save_image_tool, setup_folders_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )
