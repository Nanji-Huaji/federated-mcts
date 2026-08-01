# Few-shot logical-deduction prompts for the FOLIO dataset.
#
# Labels: "True" (conclusion follows from premises), "False" (contradicts
# premises), "Unknown" (premises give insufficient information). The model
# must learn to say "I don't know" when the premises do not determine the
# conclusion.

standard_prompt = """Determine whether the conclusion necessarily follows from the premises. Answer with the label True, False, or Unknown using the format #### [label].

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Answer: Tweety is a bird, and all birds have feathers, so Tweety has feathers. #### True

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Answer: The premises say nothing about whether birds can fly, so the conclusion cannot be determined. #### Unknown

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has no feathers.
Answer: Tweety is a bird and all birds have feathers, so Tweety has feathers, which contradicts the conclusion. #### False

{input}
Answer: """

cot_prompt = """Reason step by step about whether the conclusion necessarily follows from the premises. Deduce facts from the premises one at a time, then end with #### True, #### False, or #### Unknown.

FORMAT INSTRUCTIONS:
- State each derived fact on its own line
- Use Unknown when the premises neither prove nor disprove the conclusion
- Finish with #### [True/False/Unknown]

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Steps:
All birds have feathers.
Tweety is a bird.
Therefore, Tweety has feathers.
#### True

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Steps:
All birds have feathers.
Tweety is a bird.
Tweety has feathers.
The premises do not mention flying, so it cannot be determined.
#### Unknown

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has no feathers.
Steps:
All birds have feathers.
Tweety is a bird.
Therefore, Tweety has feathers, which contradicts the conclusion.
#### False

{input}
Steps:
"""

propose_prompt = """Given the premises, the conclusion, and the current reasoning progress, suggest the next logical deduction step.

FORMAT INSTRUCTIONS:
- State exactly one new fact that follows from the premises and prior steps
- Do not repeat facts already derived
- If the current facts already settle the conclusion, emit the final label #### True / #### False / #### Unknown

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Current progress:
All birds have feathers.
Possible next steps:
Tweety is a bird, and all birds have feathers, so Tweety has feathers.
The conclusion is settled, so the final label is #### True

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Current progress:
All birds have feathers.
Tweety is a bird.
Possible next steps:
Tweety has feathers.
The premises say nothing about flying, so the final label is #### Unknown

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has no feathers.
Current progress:
All birds have feathers.
Tweety is a bird.
Tweety has feathers.
Possible next steps:
Tweety has feathers, which contradicts the conclusion, so the final label is #### False

{input}
Current progress:
{current_progress}
Possible next steps:
"""

value_prompt = """Evaluate how close the current reasoning is to correctly determining the truth value of the conclusion (sure/likely/unlikely/impossible).

EVALUATION CRITERIA:
- Do the derived facts correctly follow from the premises?
- Does the progress clearly settle the conclusion as True, False, or Unknown?
- Is the reasoning heading toward a correct #### [label]?

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Progress:
All birds have feathers.
Tweety is a bird.
Judge: Both premises are stated; the conclusion follows directly. Ready for the final label.
sure

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Progress:
All birds have feathers.
Judge: The key premise linking Tweety to feathers is missing; one more step completes the proof.
likely

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Progress:
Tweety has feathers, so Tweety must fly.
Judge: It wrongly asserts flying follows from having feathers; flying is not entailed.
unlikely

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Progress:
Tweety has no feathers, because birds do not exist.
Judge: Contradicts the premises; the reasoning cannot be correct.
impossible

{input}
Progress:
{current_progress}
Judge:
"""

value_last_step_prompt = """Evaluate whether the given final answer correctly determines the truth value of the conclusion (sure/impossible).

EVALUATION CRITERIA:
- Do the derived facts follow from the premises?
- Is the final label correct given those facts?
- Is it formatted with #### [True/False/Unknown]?

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Answer:
All birds have feathers.
Tweety is a bird.
Tweety has feathers.
#### True
Judge: Correctly derives the conclusion and labels it True.
sure

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety can fly.
Answer:
Tweety is a bird.
Tweety has feathers.
The premises say nothing about flying.
#### Unknown
Judge: Correctly recognizes the premises are insufficient, so Unknown is right.
sure

Premises:
- All birds have feathers.
- Tweety is a bird.

Conclusion: Tweety has feathers.
Answer:
Tweety is a bird.
Tweety has feathers.
#### Unknown
Judge: The premises do entail the conclusion, so the label must be True, not Unknown.
impossible

{input}
Answer:
{answer}
Judge:
"""

joint_rank_prompt = """Rank every candidate reasoning step by how likely it is to help correctly determine the conclusion's truth value (True/False/Unknown). Return JSON only as {{"ranking":[{{"id":0,"score":0.0}}]}}. Include every ID exactly once and use scores from 0 to 1.

{input}
Candidates:
{candidates}
"""
