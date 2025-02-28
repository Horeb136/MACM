from utils.gpt_utils import local_chat_completion  

def chat_gpt(prompt, max_tokens=150, temperature=0.7, n=1):
    """
    Generate a single response using the local model.
    """
    responses = local_chat_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    return responses[0].strip()

def generate_from_GPT(prompts, max_tokens, temperature=0.7, n=3):
    """
    Generate multiple completions from the local model using the provided prompt messages.
    The function concatenates the user messages and appends an "Assistant:" marker.
    Returns a list of dictionaries mimicking the original GPT response format.
    """
    full_prompt = ""
    for message in prompts:
        full_prompt += f"User: {message['content']}\n"
    full_prompt += "Assistant: "
    
    responses = local_chat_completion(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    
    generated_ans = []
    for idx, resp in enumerate(responses):
        generated_ans.append({
            "index": idx,
            "message": {
                "role": "assistant",
                "content": resp.strip()
            },
            "finish_reason": "length"  # You can adjust this as needed
        })
    return generated_ans

def Judge_if_got_Answer_from_GPT(prompts, max_tokens, temperature=0.7, n=1):
    """
    Generate an answer from the local model using the provided prompt messages.
    """
    full_prompt = ""
    for message in prompts:
        full_prompt += f"User: {message['content']}\n"
    full_prompt += "Assistant: "
    
    responses = local_chat_completion(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    return responses[0].strip()

def Find_Answer_from_GPT(prompts, max_tokens, temperature=0.7, n=1):
    """
    Generate an answer from the local model using the provided prompt messages.
    """
    full_prompt = ""
    for message in prompts:
        full_prompt += f"User: {message['content']}\n"
    full_prompt += "Assistant: "
    
    responses = local_chat_completion(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    return responses[0].strip()
