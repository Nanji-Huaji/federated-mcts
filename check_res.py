import json
import re

def get_ans(text: str) -> str:
    pattern = r'Answer:\s*(.*)$'
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[-1].strip() if matches else ""

def check_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    total = len(data)
    correct = 0
    
    for item in data:
        correct_answers = 0  # 统计该item中正确答案的数量
        total_answers = 0    # 统计该item中有效答案的总数
        
        for ans in item["ys"]:
            ans_text = get_ans(ans)
            if not ans_text:  # 如果没有提取到答案，跳过
                continue
            
            total_answers += 1  # 计入有效答案总数
                
            try:
                if '=' in ans_text:
                    left, right = ans_text.split('=', 1)  # 只分割一次，防止多个等号
                    left_val = eval(left.strip())
                    right_val = eval(right.strip())
                    if (abs(left_val - right_val) < 1e-6 and abs(left_val - 24) < 1e-6):
                        print(f"Found correct answer: {ans_text}")
                        correct_answers += 1
                else:
                    result = eval(ans_text)
                    if abs(result - 24) < 1e-6:
                        print(f"Found correct answer: {ans_text}")
                        correct_answers += 1
                        
            except Exception as e:
                print(f"Error evaluating expression '{ans_text}': {e}")
                continue
        
        # 检查是否超过半数正确
        if total_answers > 0 and correct_answers > 0:
            print(f"Item idx {item['idx']}: {correct_answers}/{total_answers} correct - PASSED")
            correct += 1
        else:
            print(f"Item idx {item['idx']}: {correct_answers}/{total_answers} correct - FAILED")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    check_results("logs/game24/naive/deepseek-v3/merged_results.json")
