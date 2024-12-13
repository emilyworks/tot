from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Evaluation(BaseModel):
    '''
    Evaluation of a solution step.
    '''
    reasoning: str = Field(description="Reasoning behind the evaluation.")
    likelihood: Literal["sure", "likely", "unlikely"] = Field(description="The likelihood of the solution step being on the right track.")
    is_solution: bool = Field(description="Whether the solution step is the solution (final step).")


class SolutionStep(BaseModel):
    '''
    Proposal of a solution step.
    '''
    reasoning: str = Field(description="The description of the proposed solution step.")
    final: bool = Field(description="Whether the proposal is the solution (final step).")


class SimpleStep(BaseModel):
    step: str = Field(description="The reasoning for the step.")

class SimpleSolution(BaseModel):
    solution: str = Field(description="The solution to the problem.")

class SimpleSolutionChain(BaseModel):
    steps: List[SimpleStep] = Field(description="The steps to solve the problem.")
    solution: SimpleSolution = Field(description="The solution to the problem.")
