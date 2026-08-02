import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable, Tuple, List, Optional


class UncertaintyInferenceFramework:
    """
    基于transformers的推理框架，支持在生成过程中计算token级别的不确定度
    """
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        初始化推理框架
        
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名称
            device: 运行设备
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_with_uncertainty(
        self,
        input_text: str,
        uncertainty_fn: Callable[[torch.Tensor, int], float],
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[str, float, List[float]]:
        """
        生成文本并计算不确定度
        
        Args:
            input_text: 输入文本
            uncertainty_fn: 不确定度计算函数，输入为(logits, current_token)
            max_length: 最大生成长度
            temperature: 采样温度
            do_sample: 是否采样
            top_p: nucleus采样参数
            top_k: top-k采样参数
            
        Returns:
            Tuple[生成的文本, 平均不确定度, 每个token的不确定度列表]
        """
        # 编码输入文本
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        generated_tokens = inputs.clone()
        token_uncertainties = []
        
        with torch.no_grad():
            for _ in range(max_length - inputs.size(1)):
                # 获取模型输出
                outputs = self.model(generated_tokens)
                logits = outputs.logits[0, -1, :]  # 获取最后一个位置的logits
                
                # 应用温度
                sampling_logits = logits.clone()
                if temperature != 1.0:
                    sampling_logits = sampling_logits / temperature
                
                # 生成下一个token
                if do_sample:
                    # 应用top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(sampling_logits, top_k)
                        logits_filtered = torch.full_like(sampling_logits, -float('inf'))
                        logits_filtered[top_k_indices] = top_k_logits
                        sampling_logits = logits_filtered
                    
                    # 应用top-p采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(sampling_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        sampling_logits[indices_to_remove] = -float('inf')
                    
                    # 采样
                    probs = F.softmax(sampling_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心解码
                    next_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)
                
                # 计算不确定度（使用原始logits和生成的token）
                uncertainty = uncertainty_fn(logits, int(next_token.item()))
                token_uncertainties.append(uncertainty)
                
                # 添加新token
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # 检查是否生成了结束token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # 计算平均不确定度
        avg_uncertainty = sum(token_uncertainties) / len(token_uncertainties) if token_uncertainties else 0.0
        
        return generated_text, avg_uncertainty, token_uncertainties
    
    def batch_generate_with_uncertainty(
        self,
        input_texts: List[str],
        uncertainty_fn: Callable[[torch.Tensor, int], float],
        max_length: int = 100,
        **kwargs
    ) -> List[Tuple[str, float, List[float]]]:
        """
        批量生成文本并计算不确定度
        
        Args:
            input_texts: 输入文本列表
            uncertainty_fn: 不确定度计算函数，输入为(logits, current_token)
            max_length: 最大生成长度
            **kwargs: 其他生成参数
            
        Returns:
            每个输入对应的(生成文本, 平均不确定度, token不确定度列表)
        """
        results = []
        for input_text in input_texts:
            result = self.generate_with_uncertainty(
                input_text, uncertainty_fn, max_length, **kwargs
            )
            results.append(result)
        return results




# 示例不确定度计算函数
def entropy_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """
    使用熵计算不确定度
    
    Args:
        logits: 完整词表的logits，形状为(vocab_size,)
        current_token: 当前生成的token ID
        
    Returns:
        不确定度值
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    return entropy.item()


def variance_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """
    使用方差计算不确定度
    
    Args:
        logits: 完整词表的logits，形状为(vocab_size,)
        current_token: 当前生成的token ID
        
    Returns:
        不确定度值
    """
    probs = F.softmax(logits, dim=-1)
    mean = torch.sum(probs * torch.arange(len(probs), dtype=torch.float, device=probs.device))
    variance = torch.sum(probs * (torch.arange(len(probs), dtype=torch.float, device=probs.device) - mean) ** 2)
    return variance.item()


def inconsistency(logits: torch.Tensor, current_token: int) -> float:
    """
    基于温度扰动的不一致性计算不确定度
    
    Args:
        logits: 完整词表的logits，形状为(vocab_size,)
        current_token: 当前生成的token ID
        
    Returns:
        不确定度值
    """
    device = logits.device
    
    M = 20
    theta_max = 3

    # 获取原始预测（最可能的token）
    if current_token is not None:
        original_pred = current_token
    else:
        original_pred = torch.argmax(logits).item()
    
    # 从区间(0, theta_max]中采样M个温度值
    temperatures = torch.rand(M, device=device) * theta_max + 1e-6
    
    inconsistent_count = 0
    
    for m in range(M):
        # 应用温度扰动：logits / temperature
        scaled_logits = logits / temperatures[m]
        
        # 应用softmax得到概率分布
        probs = F.softmax(scaled_logits, dim=0)
        
        # 从分布中采样一个token
        sampled_token = torch.multinomial(probs, 1).item()
        
        # 检查是否与原始预测不一致
        if sampled_token != original_pred:
            inconsistent_count += 1
    
    # 返回不一致的比例
    return inconsistent_count / M


def token_probability_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """
    基于当前token概率计算不确定度
    
    Args:
        logits: 完整词表的logits，形状为(vocab_size,)
        current_token: 当前生成的token ID
        
    Returns:
        不确定度值（1 - 当前token的概率）
    """
    probs = F.softmax(logits, dim=-1)
    current_token_prob = probs[current_token].item()
    return 1.0 - current_token_prob


# Type alias for backward compatibility
DraftModel = UncertaintyInferenceFramework


# 使用示例
if __name__ == "__main__":
    # 初始化框架
    framework = UncertaintyInferenceFramework("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成文本并计算不确定度
    input_text = "The future of artificial intelligence is"
    generated_text, avg_uncertainty, token_uncertainties = framework.generate_with_uncertainty(
        input_text=input_text,
        uncertainty_fn=entropy_uncertainty,
        max_length=50,
        temperature=0.8
    )
    
    print(f"输入: {input_text}")
    print(f"生成: {generated_text}")
    print(f"平均不确定度: {avg_uncertainty:.4f}")
    print(f"每个token的不确定度: {[f'{u:.4f}' for u in token_uncertainties]}")
