# tot/prompts/aime.py

# 3-shot standard prompt
standard_prompt = '''Solve the following AIME problems step by step and provide the final numerical answer (0-999).

Problem: Find the number of ordered pairs $(a,b)$ of positive integers such that $a+b=1000$ and neither $a$ nor $b$ has a zero digit.
Answer: 738

Problem: The polynomial $P(x) = x^3 - 3x^2 + 3x - 2$ has three roots. If the roots are $r$, $s$, and $t$, find $r^2 + s^2 + t^2$.
Answer: 3

Problem: In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the length of the altitude from $A$ to $BC$.
Answer: 12

Problem: {input}
Answer: '''

# 3-shot chain-of-thought prompt
cot_prompt = '''Solve the following AIME problems with detailed step-by-step reasoning.

Problem: Find the number of ordered pairs $(a,b)$ of positive integers such that $a+b=1000$ and neither $a$ nor $b$ has a zero digit.
Solution:
Step 1: We need positive integers from 1 to 999 with no zero digits.
Step 2: For a d-digit number with no zeros, each digit can be 1,2,...,9 (9 choices).
Step 3: Count by digits: 1-digit (1-9): 9 numbers; 2-digit (10-99): 9×9=81 numbers; 3-digit (100-999): 9×9×9=729 numbers.
Step 4: Total numbers without zero digits from 1-999: 9+81+729=819.
Step 5: However, we need $a+b=1000$ where both $a$ and $b$ have no zeros.
Step 6: For each valid $a$, we need $b=1000-a$ to also be valid (no zeros).
Step 7: We must count pairs where both $a$ and $1000-a$ have no zero digits.
Step 8: By systematic counting or complementary counting, this gives us 738 valid pairs.
Answer: 738

Problem: The polynomial $P(x) = x^3 - 3x^2 + 3x - 2$ has three roots. If the roots are $r$, $s$, and $t$, find $r^2 + s^2 + t^2$.
Solution:
Step 1: Use Vieta's formulas for $P(x) = x^3 - 3x^2 + 3x - 2$.
Step 2: We have $r + s + t = 3$, $rs + rt + st = 3$, and $rst = 2$.
Step 3: We want to find $r^2 + s^2 + t^2$.
Step 4: Use the identity $(r + s + t)^2 = r^2 + s^2 + t^2 + 2(rs + rt + st)$.
Step 5: Substituting: $3^2 = r^2 + s^2 + t^2 + 2(3)$.
Step 6: Therefore: $9 = r^2 + s^2 + t^2 + 6$.
Step 7: So $r^2 + s^2 + t^2 = 9 - 6 = 3$.
Answer: 3

Problem: In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the length of the altitude from $A$ to $BC$.
Solution:
Step 1: Use Heron's formula to find the area of triangle $ABC$.
Step 2: Semi-perimeter: $s = \frac{13 + 14 + 15}{2} = 21$.
Step 3: Area = $\sqrt{s(s-a)(s-b)(s-c)} = \sqrt{21 \cdot 8 \cdot 7 \cdot 6} = \sqrt{7056} = 84$.
Step 4: The area can also be calculated as $\frac{1}{2} \cdot BC \cdot h$, where $h$ is the altitude from $A$.
Step 5: So $84 = \frac{1}{2} \cdot 14 \cdot h$.
Step 6: Solving: $h = \frac{2 \cdot 84}{14} = \frac{168}{14} = 12$.
Answer: 12

Problem: {input}
Solution:
'''

# Propose next step prompt
propose_prompt = """Continue solving this AIME problem by suggesting the next logical step in the solution process.

You should follow the format, in each step, use the format "Step X: ...", where X is the step number.

Problem: In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the length of the altitude from $A$ to $BC$.

Current problem and progress:
Step 1: Use Heron's formula to find the area of triangle $ABC$.

Next step: 
Step 2: Semi-perimeter: $s = \frac{13 + 14 + 15}{2} = 21$.

...

If you are sure this is the final step, provide the final answer in the format:
Final Answer: 12

Current problem and progress:
{input}

What should be the next step in solving this problem? Provide one clear, logical next step.
Next step: 

"""

# Value evaluation prompt for partial solutions
value_prompt = '''Evaluate the progress on this AIME problem solution. Rate the approach and current progress.

Problem: {input}

Rate this solution approach as: sure/likely/impossible

Examples:

Problem: Find the sum of all positive integers $n$ such that when $n$ is divided by 13, the remainder is 5.
Current progress: "This problem doesn't make sense because there are infinitely many such integers."
Assessment: The reasoning is flawed - we need more constraints or a finite range.
impossible

Problem: In how many ways can 8 people sit around a circular table?
Current progress: "Step 1: For circular arrangements, we fix one person to account for rotational symmetry."
Assessment: This is the correct approach for circular permutations.
sure

Problem: Find the number of integer solutions to $x^2 + y^2 = 100$.
Current progress: "Step 1: We need to find all ways to express 100 as sum of two squares."
Step 2: Since $100 = 10^2 = 2^2 \times 5^2$, we can use the factorization.
Assessment: Good start with correct factorization approach.
likely

{input}
'''

# Value evaluation for complete solutions
value_last_step_prompt = '''Evaluate if this complete AIME solution is mathematically correct and arrives at the right answer.

Problem: Find the number of ordered pairs $(a,b)$ of positive integers such that $a+b=1000$ and neither $a$ nor $b$ has a zero digit.
Solution: Step 1: Count positive integers 1-999 with no zero digits.
Step 2: 1-digit: 9, 2-digit: 81, 3-digit: 729. Total: 819.  
Step 3: Need both $a$ and $1000-a$ to have no zeros.
Step 4: By systematic counting, answer is 738.
Final Answer: 738
Judge: sure

Problem: The polynomial $P(x) = x^3 - 3x^2 + 3x - 2$ has roots $r$, $s$, $t$. Find $r^2 + s^2 + t^2$.
Solution: Step 1: By Vieta's formulas: $r+s+t=3$, $rs+rt+st=3$.
Step 2: $(r+s+t)^2 = r^2+s^2+t^2 + 2(rs+rt+st)$
Step 3: $9 = r^2+s^2+t^2 + 6$, so $r^2+s^2+t^2 = 3$.
Final Answer: 3
Judge: sure

Problem: Find the sum of all positive integers less than 1000.
Solution: Step 1: We want $1 + 2 + 3 + ... + 999$.
Step 2: Using arithmetic series formula: $\frac{n(n+1)}{2}$ with $n = 999$.
Step 3: Answer = $\frac{999 \times 1000}{2} = 499500$.
Final Answer: 499500
Judge: impossible

Problem: In triangle $ABC$ with $AB = 13$, $BC = 14$, $CA = 15$, find the altitude from $A$ to $BC$.
Solution: Step 1: Use Heron's formula with $s = 21$.
Step 2: Area = $\sqrt{21 \times 8 \times 7 \times 6} = 84$.
Step 3: Since Area = $\frac{1}{2} \times base \times height$: $84 = \frac{1}{2} \times 14 \times h$.
Step 4: Therefore $h = 12$.
Final Answer: 12
Judge: sure

Problem: {input}
Solution: {answer}
Judge: '''

# Backup prompts for different solution stages
propose_prompt_backup_s1 = '''This AIME problem is in the middle stage of solution. Suggest the next logical step to continue the mathematical reasoning.

Current state: {input}

Continue with: '''

propose_prompt_backup_s2 = '''This AIME problem is near completion. Suggest how to finalize the solution and reach the numerical answer.

Current state: {input}

Next step: '''

propose_prompt_backup_s0 = '''This AIME problem needs to be started. Suggest the first step to approach this mathematical problem.

Problem: {input}

First step: '''
