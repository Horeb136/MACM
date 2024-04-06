# MACM

## Introdcution

MACM is a system that utilizes multi agents to interact with each other in order to continuously explore potential conditions for solving complex mathematical problems.

![Basic Flow Image](Figures/Introduction.png "Basic Flow")

MACM extracts conditions and the objective from each math problem, iteratively adds new insights to the known conditions, and repeats this until enough information is gathered to reach a solution.

Compared to the old method of prompting. The advantages of MACM are as follows: 

1. **Stronger logical reasoning**. This is due to the fact that MACM removes the hierarchical structure of previous prompting methods, allowing arbitrary thoughts to be related to each other.

2. **Stronger generalization ability**. MACM does not need to re-design the prompt for each problem like the old tree of thought or graph of thought. it can be applied to arbitrary mathematical and logical reasoning problems. All the user needs to do is enter the problem and the process is completely automated.