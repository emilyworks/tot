propose_prompt = '''Given a latex problem goal and a current state, propose one possible next step to solve the problem from the current state. If the current state solves the problem, simply reiterate the current state value.
###
Here is an example to help:
Problem Goal:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Current State:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\(1/100)=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\(1/100)=26$.
Possible next step:
The difference between 36 and 26 is $\\boxed(10)$.
###
Problem Goal:
{problem}
Current State:
{current_state}
Possible next step:
'''

value_prompt = '''Given a problem goal, a current state, and a proposed next step, evaluate if the next step from the current state can help eventually solve or answer the problem (yes/likely/no)
###
Here is an example to help:
Problem Goal:
What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Current State:
One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac(1/100)=36$.
Possible next step:
$130\\%$ of 20 is $130\\cdot 20\\cdot\\frac(1/100)=26$
Evaluation:
likely
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