from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("Downloading GPT-Neo...")
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

print("Downloading NLI model...")
AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

print("Downloading embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading SQuAD datasets...")
load_dataset("squad")
load_dataset("squad_v2")

print("All models downloaded successfully.")
