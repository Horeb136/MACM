from utils.gpt_robots import generate_from_excutor
from utils.gpt import Find_Answer_from_GPT
from prompt.prompts import find_target, box_target

def Execute_steps(conditions, objectives, steps):
    """
    Uses the 'excutor' role to compute the final answer based on conditions, objectives, and prescribed steps.
    
    Input:
      - conditions (List[str]): A list of condition strings.
      - objectives (List[str]): A list of objective strings.
      - steps (str): Detailed steps to be followed.
    
    Output:
      - final answer (str): The answer produced by the model.
    """
    messages = []
    # Format conditions and objectives as numbered lists
    numbered_conditions = "\n".join(f"{i + 1}. {condition}" for i, condition in enumerate(conditions))
    numbered_objective = "\n".join(f"{i + 1}. {objective}" for i, objective in enumerate(objectives))
    
    # Create the first prompt using the 'find_target' template
    message_content = find_target.format(
        Objective=numbered_objective,
        Conditions=numbered_conditions,
        Steps=steps
    )
    messages.append({"role": "user", "content": message_content})
    
    # Append a second prompt to ask for the final boxed answer
    messages.append({"role": "user", "content": box_target})
    
    # Call the local model using our excutor function
    boxed_answer = generate_from_excutor(
        messages, 
        max_tokens=512, 
        temperature=0.7, 
        n=1
    )
    return boxed_answer

def Find_Answer(conditions, objectives):
    """
    Uses the local model to derive an answer based solely on the given conditions and objectives.
    
    Input:
      - conditions (List[str]): A list of condition strings.
      - objectives (List[str]): A list of objective strings.
    
    Output:
      - final answer (str): The answer produced by the model.
    """
    messages = []
    # Format conditions and objectives as numbered lists
    numbered_conditions = "\n".join(f"{i + 1}. {condition}" for i, condition in enumerate(conditions))
    numbered_objective = "\n".join(f"{i + 1}. {objective}" for i, objective in enumerate(objectives))
    
    # Create the prompt using the 'find_target' template (without steps)
    message_content = find_target.format(
        Objective=numbered_objective,
        Conditions=numbered_conditions
    )
    messages.append({"role": "user", "content": message_content})
    
    final_answer = Find_Answer_from_GPT(
        messages, 
        max_tokens=512, 
        temperature=0.7, 
        n=1
    )
    return final_answer

if __name__ == "__main__":
    # Example usage for testing
    test_conditions = [
        "a + b = 10",
        "a - b = 2"
    ]
    test_objectives = [
        "Find the value of a",
        "Find the value of b"
    ]
    test_steps = (
        "Step 1: Solve the system of equations.\n"
        "Step 2: Verify your solution."
    )
    
    print("Execute_steps output:")
    print(Execute_steps(test_conditions, test_objectives, test_steps))
    
    print("\nFind_Answer output:")
    print(Find_Answer(test_conditions, test_objectives))
