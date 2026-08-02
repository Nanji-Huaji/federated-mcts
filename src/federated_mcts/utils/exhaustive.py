from typing import NoReturn, TypeVar

Unhandled = TypeVar("Unhandled")


class UnhandledVariantError(RuntimeError):
    def __init__(self, value: Unhandled):
        self.value = value

    def __str__(self) -> str:
        return f"Unhandled variant: {self.value!r}"


def assert_never(value: Unhandled) -> NoReturn:
    raise UnhandledVariantError(value)
