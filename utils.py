import json
import os
import re
import shutil

def safe_json_extract(text, entity):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"1": f"{entity} in the scene (fallback)"}


def ensure_folders(output_folder, discard_folder):
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(discard_folder):
        shutil.rmtree(discard_folder)
    os.makedirs(discard_folder, exist_ok=True)
