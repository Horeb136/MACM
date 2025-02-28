# gpt_robots.py

import os
from dotenv import load_dotenv

# IMPORTANT: adjust this import path to point to your actual utils.py
from utils.gpt_utils import local_chat_completion  

load_dotenv()

def generate_from_thinker(prompts, max_tokens=512, temperature=0.7, n=1):
    """
    Mimics the 'thinker' assistant by prepending a 'thinker' instruction
    and then passing all user messages to the local DeepSeek model.
    """
    # The 'thinker' instruction
    thinker_instructions = (
        "You are a thinker. I need you to help me think about some problems. "
        "You need to provide me the answer based on the format of the example.\n"
    )
    # Build the prompt from the instructions + user messages
    prompt_text = thinker_instructions
    for message in prompts:
        prompt_text += f"User: {message['content']}\n"
    prompt_text += "Assistant: "  # marks where the model’s answer should start

    # Call your local model
    responses = local_chat_completion(
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    # Return the first response (or all, if you prefer)
    return responses[0].strip()

def generate_from_judge(prompts, max_tokens=512, temperature=0.7, n=1):
    """
    Mimics the 'judge' assistant by prepending a 'judge' instruction
    and then passing all user messages to the local DeepSeek model.
    """
    judge_instructions = (
        "You're a judge. I need you to make judgments on some statements. "
        "Be concise in your reasoning.\n"
    )
    prompt_text = judge_instructions
    for message in prompts:
        prompt_text += f"User: {message['content']}\n"
    prompt_text += "Assistant: "

    responses = local_chat_completion(
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    return responses[0].strip()

def generate_from_excutor(prompts, max_tokens=512, temperature=0.7, n=1):
    """
    Mimics the 'excutor' (executor) assistant by prepending an 'excutor' instruction
    and then passing all user messages to the local DeepSeek model.
    """
    excutor_instructions = (
        "You're an excutor. I need you to calculate the final result based on "
        "some conditions and steps. You need to provide me the answer based on "
        "the format of the examples.\n"
    )
    prompt_text = excutor_instructions
    for message in prompts:
        prompt_text += f"User: {message['content']}\n"
    prompt_text += "Assistant: "

    responses = local_chat_completion(
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    return responses[0].strip()

# Example usage:
if __name__ == "__main__":
    # Example: Thinker
    test_prompts = [{"role": "user", "content": "Explain how to find the derivative of x^2."}]
    thinker_response = generate_from_thinker(test_prompts, max_tokens=200, temperature=0.7, n=1)
    print("Thinker response:", thinker_response)

    # Example: Judge
    judge_prompts = [{"role": "user", "content": "Is the statement 'x^2 grows faster than x' correct?"}]
    judge_response = generate_from_judge(judge_prompts, max_tokens=200, temperature=0.7, n=1)
    print("Judge response:", judge_response)

    # Example: Excutor
    excutor_prompts = [{"role": "user", "content": "Compute the sum of 1+2+3+...+100."}]
    excutor_response = generate_from_excutor(excutor_prompts, max_tokens=200, temperature=0.7, n=1)
    print("Excutor response:", excutor_response)
