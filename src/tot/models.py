import os
import openai
import backoff
from collections import defaultdict

from typing import List, Tuple

import src.tot.methods.uncertainty as uncertainty
# TODO: 完成早停功能

EARLY_STOP_FUNCTION = {
    "name": "trigger_early_stop",
    "description": "当发现当前思维路径已经足够好或无需进一步探索时，触发早停机制直接跳到最终总结步骤",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "触发早停的原因，例如：'已找到最优解'、'当前路径质量足够高'、'进一步探索无意义'等"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "对当前思维质量的信心程度，0-1之间的数值"
            }
        },
        "required": ["reason", "confidence"]
    }
}

class EarlyStopException(Exception):
    """早停异常类"""
    def __init__(self, reason, confidence):
        self.reason = reason
        self.confidence = confidence
        super().__init__(f"Early stop triggered: {reason} (confidence: {confidence})")


uncertainty_calculator = uncertainty.Uncertainty(uncertainty_threshold=0.8, uncertainty_method="entropy")


# 全局模型统计字典 - 动态添加模型统计
model_usage = defaultdict(lambda: {"completion_tokens": 0, "prompt_tokens": 0, "total_calls": 0})

# 为了向后兼容，保留原有的全局变量
completion_tokens = prompt_tokens = 0
slm_completion_tokens = slm_prompt_tokens = 0
llm_completion_tokens = llm_prompt_tokens = 0

api_base = "http://127.0.0.1:11451/v1"
api_key = "lm-studio"

if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

openai.api_base = "http://127.0.0.1:11451/v1"
openai.api_key = "lm-studio"
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chatgpt(
    args,
    messages,
    model="gpt-4",
    temperature=0.5,
    max_tokens=3072,
    n=1,
    stop=None,
    api_base=openai.api_base,
    api_key=openai.api_key,
    enable_early_stop=False,  # 新增参数，控制是否启用早停功能
    **kwargs
):
    global completion_tokens, prompt_tokens, slm_completion_tokens, slm_prompt_tokens, llm_completion_tokens, llm_prompt_tokens, model_usage
    outputs = []
    
    # 构建函数调用参数
    function_params = {}
    if enable_early_stop:
        function_params["functions"] = [EARLY_STOP_FUNCTION]
        function_params["function_call"] = "auto"  # 让模型自动决定是否调用函数
    
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        
        # 构建API调用参数
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": cnt,
            "stop": stop,
            "api_base": api_base,
            "api_key": api_key,
            **function_params  # 添加函数调用参数
        }
        
        res = completions_with_backoff(**api_params)
        
        # 处理响应
        for choice in res["choices"]:
            # 检查是否有函数调用
            if "function_call" in choice["message"]:
                function_call = choice["message"]["function_call"]
                if function_call["name"] == "trigger_early_stop":
                    # 解析函数参数
                    import json
                    try:
                        params = json.loads(function_call["arguments"])
                        reason = params.get("reason", "Unknown reason")
                        confidence = params.get("confidence", 0.0)
                        # 抛出早停异常
                        raise EarlyStopException(reason, confidence)
                    except json.JSONDecodeError:
                        print("Warning: Failed to parse early stop function arguments")
            
            # 获取常规文本响应
            content = choice["message"].get("content", "")
            if content:  # 只有当有内容时才添加
                outputs.append(content)

        # 统计token使用量
        completion_tokens_used = res["usage"]["completion_tokens"]
        prompt_tokens_used = res["usage"]["prompt_tokens"]
        # 更新全局统计（向后兼容）
        completion_tokens += completion_tokens_used
        prompt_tokens += prompt_tokens_used
        # 更新模型特定统计
        model_usage[model]["completion_tokens"] += completion_tokens_used
        model_usage[model]["prompt_tokens"] += prompt_tokens_used
        model_usage[model]["total_calls"] += 1
        
        if hasattr(args, "remotebackend") and model == args.remotebackend:
            llm_completion_tokens += completion_tokens_used
            llm_prompt_tokens += prompt_tokens_used
        elif hasattr(args, "localbackend") and model == args.localbackend:
            slm_completion_tokens += completion_tokens_used
            slm_prompt_tokens += prompt_tokens_used

    if 'deepseek' in model:
        print(outputs)
        print('----------------------------')

    return outputs


def gpt(
    args,
    prompt: str,
    model: str | None = "gpt-4",
    temperature: float = 0.9,
    max_tokens: int = 1000,
    n: int = 1,
    stop=None,
    api_base: str | None = openai.api_base,
    api_key: str | None = openai.api_key,
    get_logprobs: bool = False,
    top_logprobs: int = 0,
    enable_early_stop: bool = False,  # 新增参数
    **kwargs
):
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        args,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        api_base=api_base,
        api_key=api_key,
        enable_early_stop=enable_early_stop,  # 传递早停参数
    )


def get_model_usage_summary():
    """获取所有模型的使用统计摘要"""
    summary = {}
    for model, usage in model_usage.items():
        total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
        # 根据不同模型设置不同的价格（可以根据实际情况调整）
        cost = calculate_model_cost(model, usage["completion_tokens"], usage["prompt_tokens"])

        summary[model] = {
            "completion_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "total_tokens": total_tokens,
            "total_calls": usage["total_calls"],
            "cost": cost,
        }
    return summary


def calculate_model_cost(model, completion_tokens, prompt_tokens):
    """根据不同模型计算成本"""
    # 定义不同模型的价格（每1M tokens的价格，单位：元）
    model_prices = {
        "gpt-4o": {"prompt": 2.5, "completion": 10.0},
        "gpt-4": {"prompt": 15.0, "completion": 30.0},
        "gpt-3.5-turbo": {"prompt": 0.75, "completion": 1.5},
        # 对于本地模型或其他模型，可以设置为0或很低的价格
        "meta-llama-3.1-8b-instruct@q4_k_m": {"prompt": 0.0, "completion": 0.0},
        "phi-3-medium-4k-instruct": {"prompt": 0.0, "completion": 0.0},
    }

    # 默认价格（如果模型不在列表中）
    default_price = {"prompt": 1.0, "completion": 2.0}

    # 获取模型价格，如果没有找到则使用默认价格
    price = model_prices.get(model, default_price)

    cost = completion_tokens * price["completion"] / 1000000 + prompt_tokens * price["prompt"] / 1000000

    return cost


def gpt_usage(backend="gpt-4o"):
    """获取使用统计，保持向后兼容"""
    global completion_tokens, prompt_tokens, slm_completion_tokens, slm_prompt_tokens, llm_completion_tokens, llm_prompt_tokens

    # 计算总成本
    cost = llm_completion_tokens * 10 / 1000000 + llm_prompt_tokens * 2.5 / 1000000

    # 返回原有格式的统计信息，同时添加新的模型统计
    return {
        "llm_completion_tokens": llm_completion_tokens,
        "llm_prompt_tokens": llm_prompt_tokens,
        "slm_completion_tokens": slm_completion_tokens,
        "slm_prompt_tokens": slm_prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_prompt_tokens": prompt_tokens,
        "cost": cost,
        "model_usage": get_model_usage_summary(),  # 新增：按模型的详细统计
    }


def reset_usage_stats():
    """重置所有使用统计"""
    global completion_tokens, prompt_tokens, slm_completion_tokens, slm_prompt_tokens, llm_completion_tokens, llm_prompt_tokens, model_usage

    completion_tokens = prompt_tokens = 0
    slm_completion_tokens = slm_prompt_tokens = 0
    llm_completion_tokens = llm_prompt_tokens = 0
    model_usage.clear()


def print_usage_summary():
    """打印详细的使用统计摘要"""
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)

    summary = get_model_usage_summary()
    total_cost = 0

    for model, stats in summary.items():
        print(f"\nModel: {model}")
        print(f"  Calls: {stats['total_calls']}")
        print(f"  Prompt tokens: {stats['prompt_tokens']:,}")
        print(f"  Completion tokens: {stats['completion_tokens']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Cost: ¥{stats['cost']:.4f}")
        total_cost += stats["cost"]

    print(f"\nTotal cost across all models: ¥{total_cost:.4f}")
    print("=" * 60)


# 添加一些便利函数
def get_model_list():
    """获取已使用的模型列表"""
    return list(model_usage.keys())


def get_total_tokens():
    """获取所有模型的总token数"""
    total = 0
    for usage in model_usage.values():
        total += usage["completion_tokens"] + usage["prompt_tokens"]
    return total


def get_total_cost():
    """获取所有模型的总成本"""
    total_cost = 0
    for model, usage in model_usage.items():
        total_cost += calculate_model_cost(model, usage["completion_tokens"], usage["prompt_tokens"])
    return total_cost
