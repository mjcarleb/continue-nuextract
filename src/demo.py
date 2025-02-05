import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
# Function to convert token IDs to text (Improved)
def tokens_to_text(input_ids, attention_mask):
    """
    Converts token IDs to text, handling special tokens and padding.

    Args:
        input_ids: A tensor of token IDs.
        attention_mask: A tensor indicating which tokens are real (1) and which are padding (0).

    Returns:
        A list of strings, where each string is the decoded text for a sequence in the batch.
    """

    decoded_texts = []
    for i in range(input_ids.shape[0]):  # Iterate through batches if applicable
        # Extract token IDs and attention mask for the current sequence
        sequence_ids = input_ids[i]
        sequence_mask = attention_mask[i]

        # Apply attention mask to ignore padding
        valid_indices = sequence_mask.nonzero(as_tuple=False).squeeze() # Get indices of valid tokens
        valid_ids = sequence_ids[valid_indices]

        # Decode the valid token IDs
        decoded_text = tokenizer.decode(valid_ids, skip_special_tokens=True) # Skip special tokens like [CLS], [SEP], [PAD]
        decoded_texts.append(decoded_text)

    return decoded_texts

if __name__ == "__main__":
    model_name = "numind/NuExtract-1.5"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        print(f"Model '{model_name}' loaded from cache.")

    except Exception as e:
        print(f"Error loading from cache: {e}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                      use_auth_token=HF_TOKEN,
                                                      device=device)
            model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                    use_auth_token=HF_TOKEN,
                                                                    device=device)
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

    all_inputs = []
    all_outputs = []  # Store the outputs for each chunk
        
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True) # Truncate if necessary
        all_inputs.append(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        all_outputs.append(outputs)  # Store current chunk's result
        if i > 1000:
            break


    all_predicted_texts = []

    for i, outputs in enumerate(all_outputs):  # Get the index 'i' as well
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        predicted_chunk_texts = []
        for j in range(predictions.shape[0]): # Loop through each sequence in the batch
            current_prediction = predictions[j]
            # *** KEY CHANGE: Access input_ids from the original 'inputs'
            current_input_ids = all_inputs[i].input_ids[j] # Get the input_ids for the current batch and sequence
            tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
            decoded_text = ""
            for token_index, token in enumerate(tokens):
                if token not in tokenizer.special_tokens_map.values():
                    decoded_text += token.lstrip("##")
                    if token_index < len(current_prediction): # Check if the current token has a corresponding label prediction
                        label_id = current_prediction[token_index].item()
                        label = model.config.id2label[label_id]
                        decoded_text += f"[{label}]"
            predicted_chunk_texts.append(decoded_text)
        all_predicted_texts.extend(predicted_chunk_texts)

    print(all_predicted_texts)