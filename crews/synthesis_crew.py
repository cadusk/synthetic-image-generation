import os
import time
from typing import Dict, Any, List
from PIL import Image

from crewai import Crew, Process

from agents.context_analyst import create_context_analyst_agent
from agents.image_generator import create_image_generator_agent
from agents.quality_reviewer import create_quality_reviewer_agent
from agents.data_engineer import create_data_engineer_agent
from agents.report_manager import create_report_manager_agent


class ImageSynthesisCrew:
    """
    Orchestrates the synthetic image generation pipeline using CrewAI agents.

    Pipeline flow:
    1. Setup folders (Data Engineer)
    2. For each input image:
        a. Analyze context (Context Analyst)
        b. For each context option:
            i. Generate image (Image Generator)
            ii. Review quality (Quality Reviewer)
            iii. Augment if approved (Data Engineer)
    3. Generate report (Report Manager)
    """

    def __init__(
        self,
        entity: str,
        context_limit: int,
        input_folder: str,
        output_folder: str,
        discard_folder: str,
        augment_image: bool = False
    ):
        """
        Initialize the Image Synthesis Crew.

        Args:
            entity: The entity to insert into images (e.g., 'dog', 'cat')
            context_limit: Maximum number of placement scenarios per image
            input_folder: Path to folder containing base images
            output_folder: Path to folder for approved generated images
            discard_folder: Path to folder for rejected images
            augment_image: Whether to apply data augmentation to approved images
        """
        self.entity = entity
        self.context_limit = context_limit
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.discard_folder = discard_folder
        self.augment_image = augment_image

        # Initialize agents
        self.context_analyst = create_context_analyst_agent()
        self.image_generator = create_image_generator_agent()
        self.quality_reviewer = create_quality_reviewer_agent()
        self.data_engineer = create_data_engineer_agent()
        self.report_manager = create_report_manager_agent()

        # Initialize report
        self.report = {
            "entity": entity,
            "total_images": 0,
            "api_success": 0,
            "api_failures": 0,
            "augmented_images": 0,
            "discarded": 0,
            "contexts": {}
        }

    def process_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete synthetic image generation pipeline.

        Returns:
            Dictionary containing pipeline execution report
        """
        start_time = time.time()

        # Step 1: Setup folders
        print(f"\n{'='*60}")
        print(f"Setting up folders for entity: {self.entity}")
        print(f"{'='*60}\n")
        self._setup_folders()

        # Step 2: Get list of input images
        input_images = self._get_input_images()
        print(f"Found {len(input_images)} input images to process\n")

        # Step 3: Process each input image
        for filename in input_images:
            self._process_single_image(filename)

        # Step 4: Generate report
        elapsed_time = time.time() - start_time
        self.report["processing_time"] = self._format_elapsed_time(elapsed_time)
        self._save_report()

        print(f"\n{'='*60}")
        print(f"Pipeline completed!")
        print(f"Total images processed: {self.report['total_images']}")
        print(f"Successful generations: {self.report['api_success']}")
        print(f"Failed generations: {self.report['api_failures']}")
        print(f"Discarded images: {self.report['discarded']}")
        print(f"Augmented images: {self.report['augmented_images']}")
        print(f"Processing time: {self.report['processing_time']}")
        print(f"{'='*60}\n")

        return self.report

    def _setup_folders(self):
        """Setup output and discard folders using Data Engineer agent."""
        from tools.file_tools import setup_folders_tool

        result = setup_folders_tool._run(
            output_folder=self.output_folder,
            discard_folder=self.discard_folder
        )
        print(f"✓ {result}\n")

    def _get_input_images(self) -> List[str]:
        """Get list of valid image files from input folder."""
        if not os.path.exists(self.input_folder):
            print(f"Warning: Input folder does not exist: {self.input_folder}")
            return []

        image_extensions = ('.png', '.jpg', '.jpeg')
        images = []

        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
                images.append(filename)

        return sorted(images)

    def _process_single_image(self, filename: str):
        """Process a single input image through the pipeline."""
        input_path = os.path.join(self.input_folder, filename)

        print(f"\n{'─'*60}")
        print(f"Processing: {filename}")
        print(f"{'─'*60}")

        self.report["total_images"] += 1

        # Step 1: Analyze context
        print(f"\n[1/4] Analyzing context for {self.entity} placement...")
        contexts = self._analyze_context(input_path)
        self.report["contexts"][filename] = contexts
        print(f"✓ Found {len(contexts)} placement scenarios")

        # Step 2: Process each context option
        for idx, context_description in contexts.items():
            print(f"\n[2/4] Generating image for context {idx}: '{context_description}'")
            generated_image_path = self._generate_image(input_path, context_description, idx)

            if not generated_image_path:
                print(f"✗ Image generation failed for context {idx}")
                self.report["api_failures"] += 1
                continue

            print(f"✓ Image generated successfully")
            self.report["api_success"] += 1

            # Step 3: Review quality
            print(f"[3/4] Reviewing image quality...")
            is_approved, final_path = self._review_quality(
                generated_image_path,
                filename,
                idx
            )

            if not is_approved:
                print(f"✗ Image rejected by quality review - saved to discard folder")
                self.report["discarded"] += 1
                continue

            print(f"✓ Image approved by quality review")

            # Step 4: Augment if enabled
            if self.augment_image:
                print(f"[4/4] Applying data augmentation...")
                aug_path = self._augment_image(final_path, filename, idx)
                if aug_path:
                    print(f"✓ Augmented image saved: {os.path.basename(aug_path)}")
                    self.report["augmented_images"] += 1
            else:
                print(f"[4/4] Skipping augmentation (disabled)")

        print(f"\n✓ Completed processing {filename}")

    def _analyze_context(self, image_path: str) -> Dict[str, str]:
        """Analyze image context using Context Analyst agent."""
        from tools.gemini_tools import analyze_context_tool

        contexts = analyze_context_tool._run(
            image_path=image_path,
            entity=self.entity,
            context_number=self.context_limit
        )
        return contexts

    def _generate_image(
        self,
        image_path: str,
        context_description: str,
        context_idx: str
    ) -> str:
        """Generate image with entity using Image Generator agent."""
        from tools.gemini_tools import generate_image_tool

        generated_path = generate_image_tool._run(
            image_path=image_path,
            entity=self.entity,
            context_option=context_description,
            max_retries=3
        )
        return generated_path

    def _review_quality(
        self,
        generated_image_path: str,
        base_filename: str,
        context_idx: str
    ) -> tuple:
        """Review image quality using Quality Reviewer agent."""
        from tools.gemini_tools import judge_image_tool

        # Load image and convert to bytes
        image = Image.open(generated_image_path)
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        # Judge the image
        result = judge_image_tool._run(
            image_data=image_bytes,
            entity=self.entity
        )

        is_approved = result.get("status", False)

        # Save to appropriate folder
        base_name = os.path.splitext(base_filename)[0]
        ext = os.path.splitext(base_filename)[1]

        if is_approved:
            # Save to output folder
            output_filename = f"{base_name}_ctx{context_idx}{ext}"
            final_path = os.path.join(self.output_folder, output_filename)
        else:
            # Save to discard folder
            discard_filename = f"{base_name}_ctx{context_idx}.png"
            final_path = os.path.join(self.discard_folder, discard_filename)

        image.save(final_path)
        return is_approved, final_path

    def _augment_image(
        self,
        approved_image_path: str,
        base_filename: str,
        context_idx: str
    ) -> str:
        """Augment approved image using Data Engineer agent."""
        from tools.image_tools import augment_image_tool

        base_name = os.path.splitext(base_filename)[0]
        ext = os.path.splitext(base_filename)[1]
        aug_filename = f"{base_name}_ctx{context_idx}_aug{ext}"
        aug_path = os.path.join(self.output_folder, aug_filename)

        result_path = augment_image_tool._run(
            image_path=approved_image_path,
            output_path=aug_path
        )
        return result_path

    def _save_report(self):
        """Save pipeline report using Report Manager agent."""
        from tools.file_tools import save_report_tool

        report_path = save_report_tool._run(
            report_data=self.report,
            output_folder=self.output_folder
        )
        print(f"\n✓ Report saved: {report_path}")

    def _format_elapsed_time(self, elapsed_time: float) -> str:
        """Format elapsed time as human-readable string."""
        h, rem = divmod(elapsed_time, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"
