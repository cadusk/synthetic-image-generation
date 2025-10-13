import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Synthetic Image Generation with AI and Data Augmentation + Judge")
    parser.add_argument("-e", "--entity",
                        type=str, required=True,
                        help="Entity to add in the images")

    parser.add_argument("-c", "--context_limit",
                        type=int, default=3,
                        help="The limit for generate contexts")

    parser.add_argument("-i", "--input_folder",
                        type=str, default="./images/input",
                        help="Input folder with images")

    parser.add_argument("-o", "--output_folder",
                        type=str, default="./images/output",
                        help="Output folder for generated images")

    parser.add_argument("-d", "--discard_folder",
                        type=str, default="./images/discard",
                        help="Folder for discarded images")

    parser.add_argument("-a", "--augment_image", action='store_true',
                        help="Apply additional transformations to output image")

    return parser.parse_args()

