# https://numind.ai/blog/nuextract-1-5---multilingual-infinite-context-still-small-and-better-than-gpt-4o

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import pymupdf as fitz  # Correct import
from dotenv import load_dotenv
import torch

#####################
# ENVIRONMENT
#####################
# Load environment variables from .env file
load_dotenv()

# Hugging Face credentials from .env
HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")


if not HF_USERNAME or not HF_TOKEN:
    raise ValueError("HF_USERNAME and HF_TOKEN must be set in the .env file.")

#####################
# GPU
#####################
# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS device
    print("MPS device found and will be used.")
else:
    device = torch.device("cpu") # Fallback to CPU
    print("MPS device not found. Using CPU.")

#####################
# FUNCTIONS
#####################
def predict_NuExtract(model, tokenizer, texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
    template = json.dumps(json.loads(template), indent=4)
    prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]
    
    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]

if __name__ == "__main__":
    model_name = "numind/NuExtract-1.5"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                    use_auth_token=HF_TOKEN,
                                                    device=device)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        use_auth_token=HF_TOKEN,
                                                        torch_dtype=torch.bfloat16, 
                                                        trust_remote_code=True).to(device).eval()

    except Exception as e:
        print(f"Error loading from cache: {e}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                      use_auth_token=HF_TOKEN,
                                                      device=device)
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                         use_auth_token=HF_TOKEN,
                                                         torch_dtype=torch.bfloat16, 
                                                         trust_remote_code=True).to(device).eval()
            
            print(f"Model '{model_name}' downloaded and loaded using authentication.")

        except Exception as e2:
            print(f"Error downloading/loading with authentication: {e2}")
            raise

    max_length = tokenizer.model_max_length  # Get the model's max length

    chunk_size = 2000 # Leave some space for special tokens


    # Define the path to the PDFs directory
    pdf_directory = "data/pdfs"

    # 1. PDF to Text
    pdf_full_path = os.path.join(pdf_directory, "mydoc.pdf")
    doc = fitz.open(pdf_full_path)
    text = ""
    for page in doc:
        text += page.get_text()

    text = text[:1000]

    template = """{
        "Company": {
            "Name":  "",
            "Year":  "",
            "Revenue":  ""
        }
    }"""
    
    prediction = predict_NuExtract(model, tokenizer, [text], template)[0]
    print(prediction)