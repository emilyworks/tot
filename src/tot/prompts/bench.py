propose_prompt = '''[INST]Given a problem goal and a current state, propose one possible next step to solve the problem from the current state. If the current state solves the problem, say "current state is the solution".
###
Here are two examples to help:

Example #1:
Problem Goal:
I have four numbers: 4, 4, 6, and 8. How can I use those numbers with basic arithmetic operations (+ - * /) to obtain 24?
Current State:
4+8 = 12
I have 4, 6, and 12 left.
Possible next step:
6-4 = 2

Example #2:
Problem Goal:
Choose the choice that best answer the following question:
    Question:
    Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
    Choices:
    ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
Current State:
Choice 2
Possible next step:
current state is the solution.

###
Problem Goal:
{problem}
Current State:
{current_state}
Possible next step:[\INST]
'''

value_prompt = '''[INST]Given a problem goal, a current state, and a proposed next step, evaluate if the next step from the current state can help solve or answer the problem (yes/likely/no)
###
Here are three examples to help:
Example #1:
Problem Goal: 
I have four numbers: 4, 4, 6, and 8. How can I use those numbers with basic arithmetic operations (+ - * /) to obtain 24?
Current State:
4+8 = 12 (remaining numbers: 4, 6, 12)
Proposal: 
6-4 = 2 (remaining numbers: 2, 12)
Evaluation:
likely

Example #2:
Problem Goal:
Choose the choice that best answer the following question:
    Question:
    Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
    Choices:
    ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
Current State:
Choice 4
Proposal:
current state is the solution.
Evaluation:
no

Example #3:
Choose the choice that best answer the following question:
    Question:
    Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were
    Choices:
    ['Adams only.', 'Brooks only.', 'Case only.', 'Adams and Brooks']
Current State:
Choice 2
Proposal:
current state is the solution.
Evaluation:
yes

###
Problem Goal:
{problem}
Current State:
{current_state}
Proposal:
{proposal}
Evaluation:[\INST]
'''