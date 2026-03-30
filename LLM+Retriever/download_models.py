from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch

# ---------- LLM: GPT-Neo-125M ----------
LLM_MODEL = "EleutherAI/gpt-neo-125M"  # Correct model name for GPT-Neo-125M

# 4-bit config for local download/load test
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Downloading GPT-Neo-125M tokenizer...")
tok = AutoTokenizer.from_pretrained(LLM_MODEL)

print("Downloading GPT-Neo-125M model in 4-bit...")
mdl = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",  # You can also specify device_map="cuda" if you want to force it onto GPU
)

print("Downloading NLI model...")
AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

print("Downloading embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading SQuAD datasets...")
load_dataset("squad")
load_dataset("squad_v2")

print("All models downloaded successfully.")