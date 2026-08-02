import openai
import numpy as np

class Uncertainty:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
