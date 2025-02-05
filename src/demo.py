import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import json
import pymupdf as fitz  # Correct import
from dotenv import load_dotenv
import torch

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

inputs = tokenizer(text, return_tensors="pt")

all_outputs = []  # Store the outputs for each chunk
    
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True) # Truncate if necessary
    with torch.no_grad():
        outputs = model(**inputs)
    all_outputs.append(outputs)  # Store current chunk's result
    

    a=3