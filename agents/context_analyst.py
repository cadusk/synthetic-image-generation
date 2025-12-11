import os
from crewai import Agent, LLM
from tools.gemini_tools import analyze_context_tool


def create_context_analyst_agent() -> Agent:
    """
    Creates a Context Analyst Agent responsible for analyzing images
    and identifying optimal placement scenarios for entity insertion.
    """
    # Configure Gemini LLM for the agent
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("API_KEY")
    )

    return Agent(
        role='Scene Understanding Specialist',
        goal=(
            'Analyze highway and road scene images to identify the most realistic and '
            'safe placement scenarios for inserting entities like animals, objects, or characters. '
            'Provide detailed context descriptions that enable natural-looking synthetic image generation.'
        ),
        backstory=(
            'You are an expert computer vision analyst with deep understanding of spatial '
            'relationships, object placement, and scene composition. Your specialty is analyzing '
            'road and highway environments to determine where objects can be naturally inserted. '
            'You understand perspective, scale, lighting, and environmental context. Your analysis '
            'is critical for ensuring that synthetic images look realistic and maintain proper '
            'proportions. You have years of experience in autonomous vehicle perception systems '
            'and synthetic data generation for ML training datasets.'
        ),
        tools=[analyze_context_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )
