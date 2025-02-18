import os
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")



def get_model_tokenizer(model_path) : 
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def chat_gpt(prompt):
    model, tokenizer = get_model_tokenizer(MODEL_PATH)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()


def generate_from_GPT(prompts, max_tokens, 
                      temperature=0.7, n=3):
    """
    Generate answer from GPT model with the given prompt.
    input:
        @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
        @n: the number of samples to return
    return: a list of #n generated_ans when no error occurs, otherwise None

    return example (n=3):
        [
        {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "The meaning of life is subjective and can vary greatly"
        },
        "finish_reason": "length"
        },
        {
        "index": 1,
        "message": {
            "role": "assistant",
            "content": "As an AI, I don't have personal beliefs"
        },
        "finish_reason": "length"
        },
        {
        "index": 2,
        "message": {
            "role": "assistant",
            "content": "The meaning of life is subjective and can vary greatly"
        },
        "finish_reason": "length"
        }
    ]
    """

    model, tokenizer = get_model_tokenizer(MODEL_PATH)
    responses = []
    
    for _ in range(n):  # Generate 'n' responses
        inputs = tokenizer(prompts, return_tensors="pt").to("cpu")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append({
            "index": _,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "length"
        })

    return responses

def Judge_if_got_Answer_from_GPT(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
    """
    Generate answer from GPT model with the given prompt.
    input:
        @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
        @n: the number of samples to return
    return: a list of #n generated_ans when no error occurs, otherwise None

    return example (n=3):
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompts,
            max_tokens = max_tokens,
            temperature = temperature,
            n = n
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def Find_Answer_from_GPT(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
    """
    Generate answer from GPT model with the given prompt.
    input:
        @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
        @n: the number of samples to return
    return: a list of #n generated_ans when no error occurs, otherwise None

    return example (n=3):
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompts,
            max_tokens = max_tokens,
            temperature = temperature,
            n = n
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None