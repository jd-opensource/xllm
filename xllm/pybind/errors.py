class ValidationError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: int = code
        self.message: str = message
