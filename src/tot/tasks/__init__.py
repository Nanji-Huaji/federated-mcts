def get_task(name):
    if name == "game24":
        from tot.tasks.game24 import Game24Task

        return Game24Task()
    elif name == "text":
        from tot.tasks.text import TextTask

        return TextTask()
    elif name == "crosswords":
        from tot.tasks.crosswords import MiniCrosswordsTask

        return MiniCrosswordsTask()
    elif name == "gsm8k":
        from tot.tasks.gsm8k import GSM8KTask

        return GSM8KTask()
    elif name == "aime":
        from tot.tasks.aime import AIMETask

        return AIMETask()
    elif name == "multilogieval":
        from tot.tasks.multilogieval import MultiLogicEvalTask

        return MultiLogicEvalTask()

    else:
        raise NotImplementedError
