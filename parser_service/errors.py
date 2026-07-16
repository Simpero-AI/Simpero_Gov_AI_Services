class ParseError(Exception):
    """Fail-closed parse failure, shared across parse paths (PDF, XLSX). Carries
    enough to become an HTTP error response (code/message/status_code) without
    either parser needing to import the other — the two paths are independent."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)
