import os
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

class Task:
    def __init__(self):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def get_input(self, idx: int) -> str:
        raise NotImplementedError

    def test_output(self, idx: int, output: str) -> int:
        raise NotImplementedError
