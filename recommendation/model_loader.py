from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

load_dotenv()
repo_id = "student34950242/emotion-classification"
hf_token = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    token=hf_token
)

model = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    token=hf_token
)