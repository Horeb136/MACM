# utils/utils.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load the model & tokenizer once here ---
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model = torch.compile(model) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(device)

def local_chat_completion(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    n: int = 1,
) -> list:
    """
    Generate one or more completions from the local model.
    Returns a list of response strings.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id
        )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses
