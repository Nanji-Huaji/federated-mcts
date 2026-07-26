import math
from typing import TypedDict, Callable, List, Tuple, Union, Literal

uncertainty_method = Literal["entropy", "margin", "least_confidence", "seq"]


class Uncertainty:
    def __init__(self, uncertainty_threshold: float, uncertainty_method: uncertainty_method):
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_method = uncertainty_method
        if uncertainty_method == "entropy":
            self.uncertainty_func = self._entropy
        elif uncertainty_method == "margin":
            self.uncertainty_func = self._margin
        elif uncertainty_method == "least_confidence":
            self.uncertainty_func = self._least_confidence
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

    @staticmethod
    def _entropy(logprobs: List[float]) -> float:
        """计算概率分布的熵值"""
        if not logprobs:
            return 0.0
        
        # 将logprobs转换为概率
        probs = [math.exp(logprob) for logprob in logprobs]
        
        # 计算熵
        entropy = 0.0
        for prob in probs:
            if prob > 0:
                entropy -= prob * math.log(prob)
        
        return entropy

    @staticmethod
    def _margin(logprobs: List[float]) -> float:
        """计算最高概率和次高概率之间的边距"""
        if len(logprobs) < 2:
            return 1.0
        
        # 将logprobs转换为概率并排序
        probs = sorted([math.exp(logprob) for logprob in logprobs], reverse=True)
        
        # 返回最高概率和次高概率的差值（越小越不确定）
        return probs[0] - probs[1]

    @staticmethod
    def _least_confidence(logprobs: List[float]) -> float:
        """计算最低置信度（1 - 最高概率）"""
        if not logprobs:
            return 1.0
        
        # 将logprobs转换为概率
        probs = [math.exp(logprob) for logprob in logprobs]
        max_prob = max(probs)
        
        # 返回1减去最高概率（越大越不确定）
        return 1.0 - max_prob
    
    def calculate_uncertainty(self, logprobs: List[float]) -> float:
        """计算不确定性分数"""
        return self.uncertainty_func(logprobs)
        

    def is_uncertain(self, logprobs: List[float]) -> bool:
        """根据不确定性阈值判断是否不确定"""
        uncertainty_score = self.uncertainty_func(logprobs)
        
        if self.uncertainty_method == "entropy":
            # 对于熵，值越大越不确定
            return uncertainty_score > self.uncertainty_threshold
        elif self.uncertainty_method == "margin":
            # 对于边距，值越小越不确定
            return uncertainty_score < self.uncertainty_threshold
        elif self.uncertainty_method == "least_confidence":
            # 对于最低置信度，值越大越不确定
            return uncertainty_score > self.uncertainty_threshold
        else:
            return False

    def __call__(self, logprobs: List[float]) -> bool:
        return self.is_uncertain(logprobs)