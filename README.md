# 🖼️ Synthetic Image Generation

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Google GenAI](https://img.shields.io/badge/Google-GenAI-orange)](https://ai.google.dev/)  

This project automates the generation of **synthetic images with AI** by adding custom entities (e.g., *dogs*, *cows*, etc.) into highway images, applying **data augmentation**, and filtering results with an **AI judge** to ensure quality.  
It also produces a **report** summarizing the number of images processed, successes, failures, discarded outputs, and augmented results.

---

## 📂 Project Structure
```plaintext
├── input_images/ # Folder with original images to process
├── output_images/ # Folder where accepted + augmented images are saved
│ └── <entity>/ # Entity-specific output folder (e.g., dogs, cows)
├── discarded_images/ # AI-rejected images (organized by entity)
├── .env # API key configuration
├── requirements.txt # Project dependencies
├── main.py # Main pipeline script
└── README.md # Documentation
```

---

## ⚙️ How It Works

1. **Load environment & arguments**  
   - Reads `.env` for `API_KEY`  
   - CLI args define entity, input/output folders, and number of contexts.

2. **Context Analysis (Gemini AI)**  
   - Analyzes each input image and generates JSON contexts describing **where the entity could be placed**.

3. **Entity Generation**  
   - Calls the image model to insert the chosen entity into the given context.  
   - Retries up to 3 times in case of API/server errors.

4. **Quality Judge**  
   - Another AI model evaluates the generated entity.  
   - If it looks fake → moves to `discarded_images/`.

5. **Data Augmentation**  
   - Valid images are augmented (currently with horizontal flip).  
   - Augmented versions are saved alongside the originals.

6. **Report Generation**  
   - Produces a JSON report with statistics:
     - Total images processed  
     - API successes & failures  
     - Discarded images  
     - Augmented images generated  
     - Processing time  

---

## 🖥️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-repo/synthetic-image-generation.git
cd synthetic-image-generation
pip install -r requirements.txt
```
Create a .env file in the root directory:
```bash
API_KEY=your_google_genai_api_key
```

**Usage**

Run the script with:
```bash
python main.py -e <entity> [-c CONTEXT_LIMIT] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [-d DISCARD_FOLDER]
```
Arguments

| Argument               | Description                              | Default            | Required |
| ---------------------- | ---------------------------------------- | ------------------ | -------- |
| `-e, --entity`         | Entity to insert (e.g., "dog", "cow")    | —                  | ✅ Yes    |
| `-c, --context_limit`  | Number of contexts to generate per image | `3`                | ❌ No     |
| `-i, --input_folder`   | Folder with input images                 | `input_images`     | ❌ No     |
| `-o, --output_folder`  | Folder for generated outputs             | `output_images`    | ❌ No     |
| `-d, --discard_folder` | Folder for AI-discarded results          | `discarded_images` | ❌ No     |

**Example**
```bash
python main.py -e dog -c 2 -i ./my_highways -o ./results -d ./bad_outputs
```
- Adds dogs into each highway image
- Generates up to 2 contexts per image
- Saves results in ./results/dog/
- Discards bad images into ./bad_outputs/dog/

**Report Example**

After execution, a report.json file is saved in the output entity folder:
```bash
{
  "entity": "dog",
  "total_images": 10,
  "api_success": 18,
  "api_failures": 2,
  "augmented_images": 9,
  "discarded": 4,
  "contexts": {
    "highway1.jpg": {
      "1": "dog standing on the roadside",
      "2": "dog in the middle of the road"
    }
  },
  "processing_time": "0h 3m 27s"
}
```
