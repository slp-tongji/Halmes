class StaticClass:
    def __new__(cls):
        raise RuntimeError(f"{cls.__name__} is a static class and cannot be initialized.")
    