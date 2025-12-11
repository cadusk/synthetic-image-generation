"""
Synthetic Image Generation using CrewAI Framework

This application uses a multi-agent CrewAI system to generate synthetic images
by inserting entities (animals, objects, characters) into highway/road scenes.

Pipeline:
1. Context Analyst Agent: Analyzes images to identify optimal placement scenarios
2. Image Generator Agent: Inserts entities into images using Gemini AI
3. Quality Reviewer Agent: Evaluates image quality and realism
4. Data Engineer Agent: Organizes and augments approved images
5. Report Manager Agent: Generates pipeline metrics and reports

Usage:
    python main.py --entity dog --context_limit 3 --augment_image
"""

import os
from dotenv import load_dotenv

from arguments import parse_arguments
from crews.synthesis_crew import ImageSynthesisCrew

# Load environment variables
load_dotenv()


def main():
    """
    Main entry point for the synthetic image generation pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Extract configuration
    entity = args.entity
    context_limit = args.context_limit
    input_folder = args.input_folder
    output_folder = os.path.join(args.output_folder, entity)
    discard_folder = os.path.join(args.discard_folder, entity)
    augment_image = args.augment_image

    # Validate API key
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API_KEY not found in environment variables")
        print("Please create a .env file with your Google API key:")
        print("API_KEY=your_api_key_here")
        return

    # Print configuration
    print("\n" + "="*60)
    print("SYNTHETIC IMAGE GENERATION PIPELINE")
    print("="*60)
    print(f"Entity:          {entity}")
    print(f"Context Limit:   {context_limit}")
    print(f"Input Folder:    {input_folder}")
    print(f"Output Folder:   {output_folder}")
    print(f"Discard Folder:  {discard_folder}")
    print(f"Augmentation:    {'Enabled' if augment_image else 'Disabled'}")
    print("="*60 + "\n")

    # Initialize the Image Synthesis Crew
    crew = ImageSynthesisCrew(
        entity=entity,
        context_limit=context_limit,
        input_folder=input_folder,
        output_folder=output_folder,
        discard_folder=discard_folder,
        augment_image=augment_image
    )

    # Execute the pipeline
    try:
        report = crew.process_pipeline()

        # Display summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"✓ Successfully completed synthetic image generation")
        print(f"✓ Report saved to: {output_folder}/report.json")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Pipeline execution failed")
        print(f"{'='*60}")
        print(f"Error details: {str(e)}")
        print(f"{'='*60}\n")
        raise


if __name__ == '__main__':
    main()
