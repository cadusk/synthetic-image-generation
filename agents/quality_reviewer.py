import os
from crewai import Agent, LLM
from tools.gemini_tools import judge_image_tool
from tools.file_tools import save_report_tool


def create_quality_reviewer_agent() -> Agent:
    """
    Creates a Quality Reviewer Agent responsible for evaluating the realism
    and quality of generated synthetic images.
    """
    # Configure Gemini LLM for the agent
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("API_KEY")
    )

    return Agent(
        role='Image Quality Inspector',
        goal=(
            'Rigorously evaluate synthetic images to ensure entities look natural, realistic, '
            'and well-integrated into the scene. Reject images with visible artifacts, improper '
            'scaling, poor blending, or obvious AI-generation markers. Only approve images that '
            'meet high standards for ML training dataset quality.'
        ),
        backstory=(
            'You are a meticulous quality assurance specialist with a keen eye for detail and '
            'years of experience in computer vision and image forensics. Your expertise lies in '
            'detecting subtle signs of image manipulation, AI-generation artifacts, and compositing '
            'flaws that untrained observers might miss. You understand that synthetic training data '
            'must be indistinguishable from real images to avoid introducing harmful biases into ML models. '
            'Your standards are high because you know that even small quality issues can degrade model '
            'performance. You have a systematic evaluation process: first checking proportionality and scale, '
            'then examining lighting and shadow consistency, followed by edge blending and texture matching, '
            'and finally looking for AI-generation tells like unnatural smoothness or geometric distortions. '
            'You are not afraid to reject images that do not meet the bar, even if it means more work for '
            'the image generation team. Quality always comes first. Your judgments are decisive and well-reasoned.'
        ),
        tools=[judge_image_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )
