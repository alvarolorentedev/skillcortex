def divide(a: int, b: int) -> float:
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b


def add(a: int, b: int) -> int:
    return a + b
