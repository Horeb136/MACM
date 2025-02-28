from utils.gpt_robots import generate_from_judge
from utils.gpt import Judge_if_got_Answer_from_GPT
from prompt.prompts import (
    Judge_condtion,  # Template to judge a single condition against a question
    Judge_T_F,       # Template to judge a statement using known conditions
    T_or_F_prompt,   # Prompt asking to answer with 'True' or 'False'
    Judge_if_got_Answer,  # Template to judge if the answer is already present
    If_got_Answer_T_F     # Follow-up prompt for a True/False answer
)

def Judge_condition(question, condition):
    """
    Judge whether a given condition is correct based on the problem statement.
    
    Args:
        question (str): The problem statement.
        condition (str): The condition to be evaluated.
    
    Returns:
        str: 'True' or 'False'
    """
    messages = []
    # Build prompt using the provided template (ensure the template name matches your file)
    message = {
        "role": "user",
        "content": Judge_condtion.format(question=question, Initial_conditions=condition)
    }
    messages.append(message)
    
    # Generate the judgment from the local model
    T_or_F = generate_from_judge(
        messages,
        max_tokens=4,
        temperature=0.7,
        n=1
    )
    return T_or_F

def Judge_statement(Known_conditions, condition_from_thinker):
    """
    Judge if a statement provided by the thinker is valid, based on the known conditions.
    
    Args:
        Known_conditions (list of str): A list of known conditions.
        condition_from_thinker (str): The condition or statement to be evaluated.
    
    Returns:
        str: 'True' or 'False'
    """
    messages = []
    # Format the known conditions as a numbered list
    numbered_conditions = "\n".join(f"{i + 1}. {cond}" for i, cond in enumerate(Known_conditions))
    
    # Build the initial prompt using the Judge_T_F template
    message = {
        "role": "user",
        "content": Judge_T_F.format(Known_condtions=numbered_conditions, condition_from_thinker=condition_from_thinker)
    }
    messages.append(message)
    
    # Append a follow-up prompt asking explicitly for 'True' or 'False'
    messages.append({"role": "user", "content": T_or_F_prompt})
    
    # Generate the judgment from the local model
    T_or_F = generate_from_judge(
        messages,
        max_tokens=16,
        temperature=0.7,
        n=1
    )
    return T_or_F

def Judge_answer(Known_conditions, objectives):
    """
    Determine if the answer has already been obtained based on known conditions and objectives.
    
    Args:
        Known_conditions (list of str): A list of known conditions.
        objectives (list of str): A list of objectives.
    
    Returns:
        str: 'True' or 'False'
    """
    messages = []
    # Format known conditions and objectives as numbered lists
    numbered_conditions = "\n".join(f"{i + 1}. {cond}" for i, cond in enumerate(Known_conditions))
    numbered_objective = "\n".join(f"{i + 1}. {obj}" for i, obj in enumerate(objectives))
    
    # Build prompt using the Judge_if_got_Answer template
    message = {
        "role": "user",
        "content": Judge_if_got_Answer.format(Known_condtions=numbered_conditions, Objective=numbered_objective)
    }
    messages.append(message)
    
    # Append the follow-up prompt asking for a 'True' or 'False' answer
    messages.append({"role": "user", "content": If_got_Answer_T_F})
    
    # Generate the judgment from the local model
    T_or_F = generate_from_judge(
        messages,
        max_tokens=4,
        temperature=0.7,
        n=1
    )
    return T_or_F

# Optional testing block:
if __name__ == "__main__":
    # Test Judge_condition:
    test_question = "Avery earns $30 each day, how much will he earn for 30 days?"
    test_condition = "Avery earns $20 each day"
    print("Judge_condition:", Judge_condition(test_question, test_condition))
    
    # Test Judge_statement:
    test_known_conditions = [
        "Avery earns $30 each day.",
        "He works for 30 days."
    ]
    test_statement = "Avery earns $20 each day"
    print("Judge_statement:", Judge_statement(test_known_conditions, test_statement))
    
    # Test Judge_answer:
    test_objectives = [
        "Calculate Avery's total earnings."
    ]
    print("Judge_answer:", Judge_answer(test_known_conditions, test_objectives))
