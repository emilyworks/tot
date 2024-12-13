# propose_prompt = '''Given a latex problem goal and a current state, propose one possible next step to solve the problem from the current state. Your answer should only contain one next step amd not the entire solution. If the current state solves the problem, simply reiterate the current state value.
# ###
# Here is an example to help:
# Problem Goal:
# What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
# Current State:
# One hundred twenty percent of 30 is $120\\cdot30\\cdot\\(1/100)=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\(1/100)=26$.

# Possible next step:
# The difference between 36 and 26 is $\\boxed(10)$.
# ###
# Problem Goal:
# {problem}
# Current State:
# {current_state}
# Possible next step:
# '''

solve_system_message = {
    'role': 'system',
    'content': 'You are a helpful assistant that can help solve math problems. You solve the problems step by step, with each step corresponding to a single mathematical operation. Each one of your responses is a single step. If the current step is the last step that solves the problem, then the reasoning text MUST contain \\boxed{<answer>}, while the final flag should be set to True.',
}

solve_example_1 = [
    {
        'role': 'user',
        'content': 'What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: First, we can calculate that one hundred twenty percent of 30 is $120\\cdot30\\cdot\\(1/100)=36$.\nFinal: False'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: Next, we can calculate that one hundred thirty percent of 20 is $130\\cdot 20\\cdot\\(1/100)=26$.\nFinal: False'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: Finally, we can find the difference between the two values: $36-26=\\boxed{10}$.\nFinal: True'
    }
]

solve_example_2 = [
    {
        'role': 'user',
        'content': 'How many vertical asymptotes does the graph of $y=\\frac{x-2}{x^2+x-6}$ have?'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: To find the vertical asymptotes, we can fist factor the denominator: $x^2+x-6 = (x-2)(x+3)$.\nFinal: False'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: Next, we notice that the numerator is also zero at $x=2$, so it cancels out the vertical asymptote at $x=2$.\nFinal: False'
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: Finally, we can see that there is only one vertical asymptote, at $x=-3$, so there is $\\boxed{1}$ vertical asymptote.\nFinal: True'
    }
]



value_system_message = {
    'role': 'system',
    'content': 'You are a math expert helping a student solve a math problem. Given the problem, the list of most recent previous steps, and a proposed next step, evaluate if the proposed next step is likely to help eventually solve or answer the problem (yes/likely/no).'
}


value_example_1 = [
    {
        'role': 'user',
        'content': '# Problem: What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?\n# Most recent previous steps:\nFirst, we can calculate that one hundred twenty percent of 30 is $120\\cdot30\\cdot\\(1/100)=36$\n# Proposed next step:\nNext, we can calculate that one hundred thirty percent of 20 is $130\\cdot 20\\cdot\\(1/100)=26$.',
    },
    {
        'role': 'assistant',
        'content': 'Reasoning: This step almost certainly helps solve the problem because it calculates a necessary intermediate value.\nLikelihood: sure\nIs solution: False'
    }
]

value_user_message_template = '# Problem: {problem}\n# Most recent previous steps: {steps}\n# Proposed next step: {proposal}'


simple_solve_system_message = {
    'role': 'system',
    'content': 'You are a math expert helping a student solve a math problem. Given the problem, solve it step by step, with each step corresponding to a part of the solution. The final step will be the solution to the problem.',
}

no_cot_system_message = {
    'role': 'system',
    'content': 'You are a helpful assistant.',
}

simple_solve_example_1 = [
    {
        'role': 'user',
        'content': 'What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?'
    },
    {
        'role': 'assistant',
        'content': 'Step 1: Calculate $120\\%$ of 30.\nStep 2: Calculate $130\\%$ of 20.\nStep 3: Find the positive difference between the two values.\nSolution: The positive difference between $120\\%$ of 30 and $130\\%$ of 20 is $\\boxed{10}$.'
    }
]

'''
###
Here is an example to help:

Problem Goal:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Current State:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac(1/100)=36$.
Possible next step:
$130\\%$ of 20 is $130\\cdot 20\\cdot\\frac(1/100)=26$

Evaluation:
Reasoning: This step is likely to help solve the problem.
Likelihood: likely
Is solution: False
###
Problem Goal:
{problem}
Current State:
{current_state}
Proposal:
{proposal}
Evaluation:
'''

# propose_prompt = '''Given a problem goal and a current state, propose one possible next step to solve the problem from the current state. If the current state solves the problem, say "current state is the solution".
# ###
# Here are two examples to help:

# Example #1:
# Problem Goal:
# I have four numbers: 4, 4, 6, and 8. How can I use those numbers with basic arithmetic operations (+ - * /) to obtain 24?
# Current State:
# 4+8 = 12
# I have 4, 6, and 12 left.
# Possible next step:
# 6-4 = 2

# Example #2:
# Problem Goal:
# Choose the choice that best answer the following question:
#     Question:
#     Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
#     Choices:
#     ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
# Current State:
# Brooks only.
# Possible next step:
# current state is the solution. Brooks only.

# ###
# Problem Goal:
# {problem}
# Current State:
# {current_state}
# Possible next step:
# '''

# value_prompt = '''Given a problem goal, a current state, and a proposed next step, evaluate if the next step from the current state can help solve or answer the problem (yes/likely/no)
# ###
# Here are three examples to help:
# Example #1:
# Problem Goal: 
# I have four numbers: 4, 4, 6, and 8. How can I use those numbers with basic arithmetic operations (+ - * /) to obtain 24?
# Current State:
# 4+8 = 12 (remaining numbers: 4, 6, 12)
# Proposal: 
# 6-4 = 2 (remaining numbers: 2, 12)
# Evaluation:
# likely

# Example #2:
# Problem Goal:
# Choose the choice that best answer the following question:
#     Question:
#     Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
#     Choices:
#     ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
# Current State:
# Adams and Brooks
# Proposal:
# current state is the solution. Adams and Brooks.
# Evaluation:
# no

# Example #3:
# Choose the choice that best answer the following question:
#     Question:
#     Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
#     Choices:
#     ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
# Current State:
# Brooks only.
# Proposal:
# current state is the solution. Brooks only.
# Evaluation:
# yes

# ###
# Problem Goal:
# {problem}
# Current State:
# {current_state}
# Proposal:
# {proposal}
# Evaluation:
# '''