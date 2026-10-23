import functools
from typing import Any

# Types cannot be specified; otherwise, pylance will ignore potential binding issues with `self` and `cls`.
def run_and_keep(function):    # noqa: WVarDec01
    result: Any = function()
    @functools.wraps(function)
    def wrapped():
        return result

    del function
    return wrapped
    