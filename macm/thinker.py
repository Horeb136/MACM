import re
from utils.gpt_robots import generate_from_thinker
from prompt.prompts import (
    Analysis_conditions_objective,
    Fix_conditions_prompt,
    Discover_new_conditions,
    Summarize_Answer,
    Determine_Steps
)

def Analysis_conditions(question):
    """
    Determines the conditions and objectives of a question.
    
    Args:
        question (str): The original question.
    
    Returns:
        tuple: Two lists - (conditions, objectives)
    """
    messages = []
    # Build the prompt using the template
    message = {
        "role": "user",
        "content": Analysis_conditions_objective.format(Question=question)
    }
    messages.append(message)
    
    # Generate answer using the local model
    answer = generate_from_thinker(
        messages,
        max_tokens=256,
        temperature=0.7,
        n=1
    )
    
    # Split the answer into conditions and objectives
    parts = answer.split("Objective:")
    conditions_text = parts[0].replace("Conditions:", "").strip()
    # Look for numbered conditions (e.g., "1. condition")
    conditions = re.findall(r'\d\.\s*(.*)', conditions_text)
    conditions = [condition.strip() for condition in conditions]
    
    objectives_text = parts[1].strip() if len(parts) > 1 else ""
    if re.search(r'\d\.\s+', objectives_text):
        objectives = re.findall(r'\d\.\s*(.*)', objectives_text)
    else:
        objectives = objectives_text.split('\n')
    objectives = [objective.strip() for objective in objectives if objective.strip()]
    
    return conditions, objectives


def Fix_conditions(question, Initial_conditions):
    """
    Fixes an incorrect initial condition of a question.
    
    Args:
        question (str): The original question.
        Initial_conditions (str): The wrong condition.
    
    Returns:
        str: The fixed condition.
    """
    messages = []
    message = {
        "role": "user",
        "content": Fix_conditions_prompt.format(question=question, Initial_conditions=Initial_conditions)
    }
    messages.append(message)
    
    fixed_condition = generate_from_thinker(
        messages,
        max_tokens=256,
        temperature=0.7,
        n=1
    )
    return fixed_condition


def Think_thoughts(conditions, objectives):
    """
    Asks the local model to derive new conditions based on known conditions and objectives.
    
    Args:
        conditions (list): List of conditions from Analysis_conditions.
        objectives (list): List of objectives from Analysis_conditions.
    
    Returns:
        list: A list containing the new condition(s).
    """
    messages = []
    numbered_conditions = "\n".join(f"{i + 1}. {cond}" for i, cond in enumerate(conditions))
    numbered_objective = "\n".join(f"{i + 1}. {obj}" for i, obj in enumerate(objectives))
    
    message = {
        "role": "user",
        "content": Discover_new_conditions.format(Known_conditions=numbered_conditions, Objective=numbered_objective)
    }
    messages.append(message)
    
    # Append a prompt to ask for a summary of the answer
    messages.append({"role": "user", "content": Summarize_Answer})
    
    new_condition = generate_from_thinker(
        messages,
        max_tokens=128,
        temperature=0.7,
        n=1
    )
    
    if new_condition:
        condition = [new_condition.strip()]
    else:
        condition = ["I need to rethink it"]
    return condition


def Think_Steps(condition_from_thinker, objective_from_thinker):
    """
    Asks the local model to generate the steps needed to solve the problem,
    based on the new condition(s) and objectives.
    
    Args:
        condition_from_thinker (list): List of new condition(s).
        objective_from_thinker (list): List of objectives (typically same as those from Analysis_conditions).
    
    Returns:
        str: Steps for solving the problem.
    """
    messages = []
    numbered_conditions = "\n".join(f"{i + 1}. {cond}" for i, cond in enumerate(condition_from_thinker))
    numbered_objective = "\n".join(f"{i + 1}. {obj}" for i, obj in enumerate(objective_from_thinker))
    
    message = {
        "role": "user",
        "content": Determine_Steps.format(Known_conditions=numbered_conditions, Objective=numbered_objective)
    }
    messages.append(message)
    
    steps = generate_from_thinker(
        messages,
        max_tokens=256,
        temperature=0.7,
        n=1
    )
    return steps


# Optional testing block:
if __name__ == "__main__":
    sample_question = (
        "Louis earns a base monthly salary of $1,200 with 5% commission on sales. "
        "For a month with $25,000 in sales, what are Louis's total earnings?"
    )
    conditions, objectives = Analysis_conditions(sample_question)
    print("Conditions:", conditions)
    print("Objectives:", objectives)
    
    fixed = Fix_conditions(sample_question, "Louis earns a base monthly salary of $1,000.")
    print("Fixed condition:", fixed)
    
    new_cond = Think_thoughts(conditions, objectives)
    print("New condition:", new_cond)
    
    steps = Think_Steps(new_cond, objectives)
    print("Steps:", steps)
