# PlanBench Blocksworld prompts.
#
# The problem text {input} is the canonical encoding produced by
# BlocksworldTask.get_input: it states the instance id, the blocks, the initial
# arrangement, the goal and the per-instance step bound.  The model is asked to
# emit exactly one legal action per response (no commentary), which the task's
# process_generate_result hook validates and appends to the trajectory.

standard_prompt = """You are solving a Blocksworld planning problem. Output the complete plan, one action per line, with no commentary.

{input}
Plan:
"""

cot_prompt = """You are solving a Blocksworld planning problem. Reason step by step about the current arrangement, then output the complete plan, one action per line, with no commentary.

{input}
Plan:
"""

propose_prompt = """You are solving a Blocksworld planning problem. The current arrangement of blocks is shown below. Output exactly one legal action that makes progress toward the goal, as a single line, with no commentary.

The only legal actions are:
- pick-up X
- put-down X
- stack X Y
- unstack X Y

A pick-up or unstack requires the hand to be empty; a stack requires holding the block and a clear target; an unstack requires the block to be on top of the stated block and clear.

{input}

Current state:
{current_state}

Next action (exactly one line):
"""

value_prompt = """Evaluate how close the current Blocksworld arrangement is to the goal (sure/likely/unlikely/impossible).

EVALUATION CRITERIA:
- Does the current state satisfy the goal? Answer: sure.
- Do all moves so far stay legal and does the state move toward the goal? Answer: likely.
- Are the recent moves heading away from the goal or wasting steps? Answer: unlikely.
- Is the current state invalid, or is the goal unreachable within the step limit? Answer: impossible.

{input}

Current state:
{current_state}

Goal:
{goal}

Judge:
"""

joint_rank_prompt = """Rank every candidate Blocksworld state by how likely it is to reach the goal within the step limit. Return JSON only as {{"ranking":[{{"id":0,"score":0.0}}]}}. Include every ID exactly once and use scores from 0 to 1.

{input}
Candidates:
{candidates}
"""
