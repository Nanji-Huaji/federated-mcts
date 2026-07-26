def get_task(name):
    if name == 'game24':
        from federated_mcts.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from federated_mcts.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from federated_mcts.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    elif name == 'gsm8k':
        from federated_mcts.tasks.gsm8k import GSM8KTask
        return GSM8KTask()
    else:
        raise NotImplementedError
