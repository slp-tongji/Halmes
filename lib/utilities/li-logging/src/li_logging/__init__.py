def _():
    from pathlib import Path
    import traceback
    from typing import Any

    import tomlkit
    import tomlkit.items
    import tomlkit.exceptions

    @staticmethod
    @tomlkit.register_encoder
    def _custom_encoder(obj: Any, 
                        _parent: tomlkit.items.Item | None = None, 
                        _sort_keys: bool = False):
        if obj is None:
            return tomlkit.string("None")
        if isinstance(obj, Path):
            return tomlkit.string(str(obj))
        if isinstance(obj, set):
            return tomlkit.item(list(obj), _parent, _sort_keys)
        if isinstance(obj, BaseException):
            return tomlkit.string("\n".join(traceback.format_exception(obj)),
                                literal=True, multiline=True)
        raise tomlkit.exceptions.ConvertError(f"Unsupported type '{type(obj)}'.")
_()


from li_logging.fake_toml_log import FakeTomlLog as FakeTomlLog
from li_logging.toml_log_like import TomlLogLike as TomlLogLike
from li_logging.toml_log import TomlLog as TomlLog
from li_logging.toml_logger import TomlLogger as TomlLogger
