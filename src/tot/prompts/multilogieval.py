# 5-shot
standard_prompt = """Analyze the logical context and answer the question with 'yes' or 'no'.

Context: If the factory follows environmental regulations, pollution levels will be low. If they increase production, more waste is generated. Either the factory follows regulations, or waste levels did not rise. If pollution is low, air quality will be good. If air quality is good, there will be fewer illnesses.
Question: If the factory increased production, did illnesses decrease in the town?
Answer: yes

Context: If the team practices hard, they will be prepared for the big game. If the star player is injured, he cannot play in the game. Either the team practiced hard, or the star player is not injured. When the team is well-prepared, the team wins the important game. If the team wins the important game, they will celebrate with a team dinner afterwards.
Question: If the star player is injured, will the team celebrate with a team dinner after the big game?
Answer: yes

Context: If the farmer waters his crops daily, they will grow strong and healthy. If there is a drought, then the crops will dry up. Either the farmer waters the crops routinely, or the crops did not dry up. If the crops grow strong, there will be a bountiful harvest. If there is a large harvest, the farmer will make good profits.
Question: If there was a drought this season, will the farmer make good profits?
Answer: yes

Context: If it rains, the ground will be wet. If the ground is wet, people will use umbrellas. Either it doesn't rain, or people forgot their umbrellas at home. If people use umbrellas, they will stay dry. If people stay dry, they will be comfortable.
Question: If it rained today, were people comfortable?
Answer: no

Context: If the student studies hard, they will pass the exam. If they pass the exam, they will graduate. Either the student didn't study hard, or they failed the exam. If they graduate, they will get a job. If they get a job, they will be financially stable.
Question: If the student studied hard, will they be financially stable?
Answer: no

{input}
Answer: """

# 5-shot
cot_prompt = """Analyze the logical context step by step and answer the question with 'yes' or 'no'.

Context: If the factory follows environmental regulations, pollution levels will be low. If they increase production, more waste is generated. Either the factory follows regulations, or waste levels did not rise. If pollution is low, air quality will be good. If air quality is good, there will be fewer illnesses.
Question: If the factory increased production, did illnesses decrease in the town?
Reasoning:
Step 1: Given "factory increased production" and "If they increase production, more waste is generated", waste levels rose.
Step 2: From "Either the factory follows regulations, or waste levels did not rise" and waste levels rose, the factory doesn't follow regulations.
Step 3: From "If the factory follows environmental regulations, pollution levels will be low" (contrapositive), if factory doesn't follow regulations, pollution levels are not low.
Step 4: From "If pollution is low, air quality will be good" (contrapositive), if pollution is not low, air quality is not good.
Step 5: From "If air quality is good, there will be fewer illnesses" (contrapositive), if air quality is not good, there will not be fewer illnesses (more illnesses).
Answer: yes

Context: If the team practices hard, they will be prepared for the big game. If the star player is injured, he cannot play in the game. Either the team practiced hard, or the star player is not injured. When the team is well-prepared, the team wins the important game. If the team wins the important game, they will celebrate with a team dinner afterwards.
Question: If the star player is injured, will the team celebrate with a team dinner after the big game?
Reasoning:
Step 1: Given "star player is injured" and "If the star player is injured, he cannot play in the game", the star player cannot play.
Step 2: From "Either the team practiced hard, or the star player is not injured" and star player is injured, the team practiced hard.
Step 3: From "If the team practices hard, they will be prepared for the big game" and team practiced hard, the team is prepared.
Step 4: From "When the team is well-prepared, the team wins the important game" and team is prepared, the team wins.
Step 5: From "If the team wins the important game, they will celebrate with a team dinner afterwards" and team wins, they celebrate with dinner.
Answer: yes


Context: If it rains, the ground will be wet. If the ground is wet, people will use umbrellas. Either it doesn't rain, or people forgot their umbrellas at home. If people use umbrellas, they will stay dry. If people stay dry, they will be comfortable.
Question: If it rained today, were people comfortable?
Reasoning:
Step 1: Given "it rained" and "If it rains, the ground will be wet", the ground is wet.
Step 2: From "If the ground is wet, people will use umbrellas" and ground is wet, people will use umbrellas.
Step 3: From "Either it doesn't rain, or people forgot their umbrellas at home" and it rained, people forgot their umbrellas at home.
Step 4: People need umbrellas but forgot them at home, so they cannot use umbrellas.
Step 5: From "If people use umbrellas, they will stay dry" (contrapositive), if people don't use umbrellas, they won't stay dry.
Step 6: From "If people stay dry, they will be comfortable" (contrapositive), if people don't stay dry, they won't be comfortable.
Answer: no


{input}
Reasoning:
"""

# 1-shot
propose_prompt = """Continue the logical reasoning by proposing the next step. Please follow the format below. You should only suggest multiple possible next steps based on the given step, without any further reasoning. For example, if the given step is Step 2, you only need to suggest possible Step 3 options.

Context: If A then B. If B then C. Either not A or not C. A is true.
Question: Is C true?
Current reasoning:
Step 1: Given A is true and "If A then B", we can deduce B is true.

Possible next steps:
Step 2: From "If B then C" and B is true, we can deduce C is true.
Step 2: From "Either not A or not C" and A is true, we can deduce not C (C is false).
Step 2: We need to check the consistency of our deductions so far.

Current reasoning:
{input}

Possible next steps:
"""

value_prompt = """Evaluate the logical reasoning progress (impossible/unlikely/likely/sure).

Context: If A then B. If B then C. A is true.
Question: Is C true?
Current reasoning:
Step 1: Given A is true and "If A then B", we can deduce B is true.
Evaluation: This step correctly applies modus ponens. The reasoning is sound so far.
likely

Context: If X then Y. Either not X or Z. X is true.  
Question: Is Z true?
Current reasoning:
Step 1: Given X is true and "If X then Y", we can deduce Y is true.
Step 2: From "Either not X or Z" and X is true, we can deduce Z is true.
Evaluation: Both steps are logically valid. The reasoning correctly leads to the conclusion.
sure

Context: If P then Q. If Q then R. P is false.
Question: Is R true?
Current reasoning:
Step 1: Since P is false, we cannot conclude anything about Q from "If P then Q".
Evaluation: This is correct - we cannot affirm the consequent when the antecedent is false.
likely

Context: If A then B. If C then D. A is true.
Question: Is D true?
Current reasoning:
Step 1: Given A is true and "If A then B", we can deduce B is true.
Step 2: We need to find a connection between B and D to answer the question.
Evaluation: No direct logical path from the premises to D. This reasoning cannot reach the answer.
impossible

Context: If X then Y. If Y then Z. X is true.
Question: Is Z false?
Current reasoning:
Step 1: Given X is true and "If X then Y", we can deduce Y is true.
Step 2: From "If Y then Z" and Y is true, we can deduce Z is true.
Step 3: Therefore Z is true, not false.
Evaluation: The logical chain is valid but contradicts the question asking if Z is false.
sure

{input}
Current reasoning:
{current_reasoning}
Evaluation: """

value_final_prompt = """Evaluate if the logical reasoning and final answer are correct (impossible/sure).

Context: If A then B. If B then C. A is true.
Question: Is C true?
Reasoning and Answer:
Step 1: Given A is true and "If A then B", we can deduce B is true.
Step 2: From "If B then C" and B is true, we can deduce C is true.
Answer: yes
Judge: The reasoning correctly applies logical rules and reaches the right conclusion.
sure

Context: If X then Y. Either not X or Z. X is true.
Question: Is Y true?
Reasoning and Answer:
Step 1: Given X is true and "If X then Y", we can deduce Y is true.
Answer: yes
Judge: Direct application of modus ponens. Correct reasoning and answer.
sure

Context: If P then Q. P is false.
Question: Is Q true?
Reasoning and Answer:
Step 1: Given P is false and "If P then Q", we can deduce Q is false.
Answer: no
Judge: This commits the fallacy of denying the antecedent. Cannot conclude Q is false.
impossible

Context: If A then B. If C then D. A is true.
Question: Is D true?
Reasoning and Answer:
Step 1: Given A is true and "If A then B", we can deduce B is true.
Step 2: There is no logical connection between B and D.
Answer: no
Judge: While the reasoning about A and B is correct, there's insufficient information about D.
impossible

Context: If X then Y. If Y then Z. X is true.
Question: Is Z true?
Reasoning and Answer:
Step 1: Given X is true and "If X then Y", we can deduce Y is true.
Step 2: From "If Y then Z" and Y is true, we can deduce Z is true.
Answer: yes
Judge: Perfect logical chain from X to Y to Z. Reasoning and answer are both correct.
sure

{input}
Reasoning and Answer:
{reasoning}
Judge: """

final_answer_prompt = """Based on the logical reasoning steps, provide the final answer.

{input}

Reasoning steps:
{steps}

Answer: """
