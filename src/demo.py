import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face credentials from .env
HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_USERNAME or not HF_TOKEN:
    raise ValueError("HF_USERNAME and HF_TOKEN must be set in the .env file.")

model_name = "numind/NuExtract-1.5"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    print(f"Model '{model_name}' loaded from cache.")

except Exception as e:
    print(f"Error loading from cache: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        print(f"Model '{model_name}' downloaded and loaded using authentication.")

    except Exception as e2:
        print(f"Error downloading/loading with authentication: {e2}")
        raise
                                             
# Define the path to the PDFs directory
pdf_directory = "data/pdfs"

# Iterate through each PDF file in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        # Load the PDF file
        pdf_path = os.path.join(pdf_directory, filename)
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()

        # Generate a prompt for the model
        template = json.dumps(json.loads(template), indent=4)
        prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in pdf_data.split()]

        # Generate JSON pairs from the extracted text
        outputs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)
                pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
                outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Save the extracted JSON pairs to a file
        output_filename = os.path.splitext(filename)[0] + "_extracted.json"
        with open(os.path.join("data/annotations.json", output_filename), 'w') as f:
            json.dump(outputs, f)