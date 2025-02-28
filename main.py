import os
import re
import json
import random
from collections import Counter

# Import all prompt templates from your prompts file
from prompt.prompts import *

# Import local agent functions from your project modules.
# Adjust these import paths if your project structure is different.
from macm.executor import Execute_steps
from macm.judge import Judge_statement, Judge_answer, Judge_condition
from macm.thinker import Analysis_conditions, Think_thoughts, Think_Steps


def check_condition(question, condition, n):
    """
    Use several Judges to check a single condition against the question.
    
    Args:
        question (str): The problem statement.
        condition (str): The condition to be evaluated.
        n (int): The number of evaluation rounds.
    
    Returns:
        bool: True if the condition passes in all rounds, else False.
    """
    for _ in range(n):
        if Judge_condition(question, condition).strip() == "False":
            return False
    return True


def check_statement(conditions, statement, n):
    """
    Use several Judges to check a statement against the given conditions.
    
    Args:
        conditions (list of str): Known conditions.
        statement (str): The statement to be evaluated.
        n (int): Number of evaluation rounds.
    
    Returns:
        bool: True if the statement is accepted in all rounds, else False.
    """
    for _ in range(n):
        answer = Judge_statement(conditions, statement)
        if "False" in answer or "false" in answer:
            return False
    return True


def check_answer(conditions, statement):
    """
    Use a Judge to check if an answer is already obtained based on conditions.
    
    Args:
        conditions (list of str): Known conditions.
        statement (str): The objectives (or answer) to verify.
    
    Returns:
        bool: True if the answer is accepted, else False.
    """
    if_got_answer = Judge_answer(conditions, statement)
    if "False" in if_got_answer or "false" in if_got_answer:
        return False
    return True


def check_if_got_answer(conditions, statement, n):
    """
    Repeats answer-checking n times.
    
    Args:
        conditions (list of str): Known conditions.
        statement (str): The objectives.
        n (int): Number of evaluation rounds.
    
    Returns:
        bool: True if all rounds confirm the answer, else False.
    """
    for _ in range(n):
        if not check_answer(conditions, statement):
            return False
    return True    


def main(question, times, n, min_voters, max_voters):
    """
    Uses a multi-agent process to extract conditions, generate new thoughts,
    evaluate them with judges, and finally compute an answer.
    
    Args:
        question (str): The problem statement.
        times (int): Upper limit on iterations for generating new conditions.
        n (int): Number of verification rounds for each judge check.
        min_voters (int): Minimum number of thinker-voters.
        max_voters (int): Maximum number of thinker-voters.
    
    Returns:
        str: The final answer as determined by the majority vote.
    """
    possible_answers = []
    try:
        voter_count = 0
        tie = True

        # Voting loop: add more "voters" until a consensus is reached or max_voters is hit.
        while tie or voter_count < min_voters:
            voter_count += 1
            print(f"\n# {voter_count} Thinker is analyzing the question...")
            conditions, objectives = Analysis_conditions(question)
            initial_condition_numbers = len(conditions)
            
            # Iterate to mine new conditions (but do not exceed the defined times)
            for time in range(times):
                print(f"\n# {voter_count} Thinker is thinking new thoughts...")
                unchecked_conditions = Think_thoughts(conditions, objectives)
                checked_conditions = []
                for unchecked_condition in unchecked_conditions:
                    print(f"\n# {voter_count} Judge is checking conditions...")
                    if check_statement(conditions, unchecked_condition, n):
                        # Optionally strip the prompt's explanation.
                        start = unchecked_condition.find("we can get: ")
                        if start != -1:
                            unchecked_condition = unchecked_condition[start + len("we can get: "):]
                            unchecked_condition = unchecked_condition.split("Reason:")[0]
                        checked_conditions.append(unchecked_condition)
                conditions = conditions + checked_conditions

                # If the judges agree that the objectives have been reached, break out.
                if check_if_got_answer(conditions, objectives, 1):
                    break

            print(f"\n# {voter_count} Thinker is generating solution steps...")
            steps = Think_Steps(conditions, objectives)
            
            print(f"\n# {voter_count} Executor is computing the answer...")
            final_answer = Execute_steps(conditions, objectives, steps)
            
            # Extract the answer inside the \boxed{...} marker.
            Answer_match = re.search(r'\\boxed\{(.*?)(?=\})', final_answer)
            if Answer_match:
                Answer_boxed = Answer_match.group(1).strip()
            else:
                Answer_boxed = "No match found"
            possible_answers.append(Answer_boxed)

            if voter_count >= min_voters:
                counter = Counter(possible_answers)
                most_votes = counter.most_common(1)[0][1]
                tie_count = len([item for item in counter.items() if item[1] == most_votes])
                tie = tie_count > 1

                if tie:
                    print("\nThere is a tie vote. Adding another voter...")
                if voter_count >= max_voters:
                    print("\nReached maximum voter limit.")
                    break

        most_possible_answer, count = Counter(possible_answers).most_common(1)[0]
        print(f"\nThe final answer is: {most_possible_answer}")
        return most_possible_answer
    except Exception as e:
        print(f"Error processing file: {e}")


def evaluate_dataset(folder_path, times, n, limit=5):
    """
    Evaluate a dataset of JSON files containing problems and solutions.
    
    Args:
        folder_path (str): The folder path containing JSON files.
        times (int): The upper limit for mining new conditions.
        n (int): The number of verification rounds for judge checks.
        limit (int): How many files to process.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    random.shuffle(all_files)  # Shuffle the file order

    for count, file_path in enumerate(all_files[:limit]):
        with open(file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                problem = data.get("problem")
                if problem:
                    print(f"\n# {count} Problem:\n{problem}")
                    solution = data.get("solution")
                    print(f"\n# {count} Provided Solution:\n{solution}")
                    main(problem, times, n, min_voters=5, max_voters=7)
            except json.JSONDecodeError:
                print(f"Error reading file {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    # Set verification parameters
    n = 1            # Number of verification rounds per judge check
    times = 5        # Maximum iterations for mining new conditions
    min_voters = 5   # Minimum number of thinker-voters required
    max_voters = 7   # Maximum number of thinker-voters allowed
    question = ""    # Set your own question here

    # To evaluate a dataset, use evaluate_dataset(folder_path, times, n, limit)
    # Otherwise, to process a single question, call main(question, times, n, min_voters, max_voters)
    main(question, times, n, min_voters, max_voters)
