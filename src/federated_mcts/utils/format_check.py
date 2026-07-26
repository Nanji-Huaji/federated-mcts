from collections import Counter
import re
import math

def check_and_fix_last_line(new_proposal: str, x: str = "") -> (bool, str):
    # Get the last two lines 获取最后的2次操作
    lines = new_proposal.strip().split('\n')
    last_line = lines[-1]

    if len(lines) < 2: # First generation: less than two operations
        pre_left_num = x.strip().split() 
    else:
        pre_line = lines[-2]
        pre_left_num = pre_line.split('left: ')[-1].split(')')[0].strip().split()
    # Count occurrences of each number in pre_left_num
    contains_str = all(item.isdigit() for item in pre_left_num)
    if contains_str==False:
        return False, new_proposal
    pre_left_num_counts = Counter(pre_left_num)  # Count occurrences of each number

    # Check the format of last_line 检查格式
    if '(left:' not in last_line:
        #print("Invalid format. Unable to check.")
        return False, new_proposal

    # Extract the equation part and left part 截取算术部分和剩余数字部分
    equation_part = last_line.split('(left: ')[0].strip()
    last_left_num = last_line.split('left: ')[-1].split(')')[0].strip().split()

    # Check the equation format 抽取出运算符和两个数字
    try:
        if '+' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('+'))
            operator = '+'
        elif '-' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('-'))
            operator = '-'
        elif '*' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('*'))
            operator = '*'
        elif '/' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('/'))
            operator = '/'
        else:
            #print("Unsupported operator.")
            return False, new_proposal

        # Extract the result from the equation
        result = int(equation_part.split('=')[-1].strip())
    except ValueError:
        #print("Equation format error.")
        return False, new_proposal

    # Check if a and b are in pre_left_num, and if a == b, ensure at least two occurrences 检查两个数字是否在上一次的left里
    if (str(a) not in pre_left_num_counts or 
        str(b) not in pre_left_num_counts or 
        (a == b and pre_left_num_counts[str(a)] < 2)):
        #print("Invalid number selection from pre_left_num.")
        return False, new_proposal

    # Calculate the correct result 获取正确的结果
    if operator == '+':
        correct_result = a + b
    elif operator == '-':
        correct_result = a - b
    elif operator == '*':
        correct_result = a * b
    elif operator == '/':
        if b == 0:
            #print("Division by zero error.")
            return False, new_proposal
        correct_result = a // b  # Use integer division
    else:
        return False, new_proposal

    # Check if the calculated result matches the given result  检查计算结果是否正确
    false_result = False
    if correct_result != result:
        result = correct_result
        false_result = True
    full_pattern = r'^\d+ [\+\-\*/] \d+ = \d+ \(left: .+\)$' # Check the format
    if re.match(full_pattern, last_line)==None:
        false_result = True
    if(operator == '+' and a>b or operator == '*' and a>b):
        false_result=True
        a, b = b, a


    # Create the correct left numbers by removing a and b from pre_left_num and adding the result 获取正确的left
    pre_left_list = pre_left_num.copy()
    pre_left_list.remove(str(a))
    pre_left_list.remove(str(b))
    pre_left_list.append(str(result))

    # Sort and form the new left numbers
    new_left_num = sorted(pre_left_list, key=int)
    int_left_num =list(map(int, pre_left_list))
    product = math.prod(int_left_num)
    if(product<24):
        return False, new_proposal

    
    # Check if last_left_num matches the expected new_left_num
    if ',' in ' '.join(last_left_num) or new_left_num != last_left_num or false_result:
        updated_left_part = ' '.join(new_left_num)
        updated_last_line = f"{a} {operator} {b} = {result} (left: {updated_left_part})"

        # Update new_proposal with the corrected last line
        #print("The left part is incorrect. Fixing the left part.", updated_last_line)

        if len(lines) < 2:
            updated_new_proposal = updated_last_line + '\n'
        else:
            updated_new_proposal = '\n'.join(lines[:-1]) + '\n' + updated_last_line + '\n'
        return True, updated_new_proposal

    return True, new_proposal

def check_final_result(last_line: str, pre_line: str = "", x: str = "") -> (bool, str):
    # Get the last two lines 获取最后的2次操作
    # lines = new_proposal.strip().split('\n')
    # last_line = lines[-1]

    if x != "": # First generation: less than two operations
        pre_left_num = x.strip().split() 
    else:
        pre_left_num = pre_line.split('left: ')[-1].split(')')[0].strip().split()
    contains_str = all(item.isdigit() for item in pre_left_num)
    if contains_str==False:
        return False, last_line
    # Count occurrences of each number in pre_left_num
    pre_left_num_counts = Counter(pre_left_num)  # Count occurrences of each number

    # Check the format of last_line 检查格式
    if '(left:' not in last_line:
        #print("Invalid format. Unable to check.")
        return False, last_line

    # Extract the equation part and left part 截取算术部分和剩余数字部分
    equation_part = last_line.split('(left: ')[0].strip()
    last_left_num = last_line.split('left: ')[-1].split(')')[0].strip().split()

    # Check the equation format 抽取出运算符和两个数字
    try:
        if '+' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('+'))
            operator = '+'
        elif '-' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('-'))
            operator = '-'
        elif '*' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('*'))
            operator = '*'
        elif '/' in equation_part:
            a, b = map(int, equation_part.split('=')[0].split('/'))
            operator = '/'
        else:
            #print("Unsupported operator.")
            return False, last_line

        # Extract the result from the equation
        result = int(equation_part.split('=')[-1].strip())
    except ValueError:
        #print("Equation format error.")
        return False, last_line

    # Check if a and b are in pre_left_num, and if a == b, ensure at least two occurrences 检查两个数字是否在上一次的left里
    if (str(a) not in pre_left_num_counts or 
        str(b) not in pre_left_num_counts or 
        (a == b and pre_left_num_counts[str(a)] < 2)):
        #print("Invalid number selection from pre_left_num.")
        return False, last_line

    # Calculate the correct result 获取正确的结果
    if operator == '+':
        correct_result = a + b
    elif operator == '-':
        correct_result = a - b
    elif operator == '*':
        correct_result = a * b
    elif operator == '/':
        if b == 0:
            #print("Division by zero error.")
            return False, last_line
        correct_result = a // b  # Use integer division
    else:
        return False, last_line

    # Check if the calculated result matches the given result  检查计算结果是否正确
    false_result = False
    if correct_result != result:
        result = correct_result
        false_result = True
        return False, last_line
    full_pattern = r'^\d+ [\+\-\*/] \d+ = \d+ \(left: .+\)$' # Check the format
    if re.match(full_pattern, last_line)==None:
        false_result = True
    if(operator == '+' and a>b or operator == '*' and a>b):
        false_result=True
        a, b = b, a


    # Create the correct left numbers by removing a and b from pre_left_num and adding the result 获取正确的left
    pre_left_list = pre_left_num.copy()
    pre_left_list.remove(str(a))
    pre_left_list.remove(str(b))
    pre_left_list.append(str(result))

    # Sort and form the new left numbers
    new_left_num = sorted(pre_left_list, key=int)
    
    # Check if last_left_num matches the expected new_left_num
    if new_left_num != last_left_num:
        return False, last_line
    if ',' in ' '.join(last_left_num) or false_result:
        #print("The left part is incorrect. Fixing the left part.")
        updated_left_part = ' '.join(new_left_num)
        updated_last_line = f"{a} {operator} {b} = {result} (left: {updated_left_part})"

        # Update new_proposal with the corrected last line
        #updated_new_proposal = '\n'.join(lines[:-1]) + '\n' + updated_last_line + '\n'
        return True, updated_last_line

    return True, last_line

# Example test
if __name__ == '__main__':
    # new_proposal = "3 * 1 = 3 (left: 1 3 8)\n1 + 3 = 5 (left: 4, 8)\n\n"
    # print("Original new_proposal:")
    # print(new_proposal)
    # is_correct, updated_new_proposal = check_and_fix_last_line(new_proposal, '')
    # print("Is the last line correct:", is_correct)
    # print("Updated new_proposal:")
    # print(updated_new_proposal)

    x='1 1 4 6'
    final = '\n4 - 1 = 2 (left: 1 3 6)\n3 +6 = 9 (left: 1 9)\n\n1 * 9 = 9 (left: 9)\n\n'
    final = final.split('\n')
    final_list = list(filter(None, final))
    print(final_list)
    for idx, line in enumerate(final_list):
        if(idx==0): 
            a,b = check_final_result(line, x=x)
        else:
            a,b = check_final_result(line, final_list[idx-1])
        print(a, b)
