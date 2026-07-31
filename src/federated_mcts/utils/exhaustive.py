from typing import NoReturn


class UnhandledVariantError(RuntimeError):
    def __init__(self, value: NoReturn):
        self.value = value

    def __str__(self) -> str:
        return f"Unhandled variant: {self.value!r}"


def assert_never(value: NoReturn) -> NoReturn:
    raise UnhandledVariantError(value)
