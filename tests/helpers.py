from contextlib import contextmanager
import traceback


def close_enough(x, y, tol=1e-3):
    return abs(x - y) / y < tol


@contextmanager
def assert_all():
    errors = []
    try:
        yield errors
    finally:
        errors = [error for error in errors if error != "Passed"]
        if errors:
            raise AssertionError(f"Multiple Failures:\n{''.join(errors)}")


def soft_compare(value, target, error_message, **kwargs):
    if not close_enough(value, target, **kwargs):
        return f"Assertion failed: {error_message}\n val:{value} != target:{target}\n"
    return "Passed"


def soft_assert(condition, error_message):
    if not condition:
        return f"Assertion failed: {error_message}"
    return "Passed"
