"""Domain errors carry a stable, machine-readable ``code``."""


class DomainError(Exception):
    """Base class: ``code`` is stable, ``message`` is for humans."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidDuration(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_duration", message)


class InvalidSetting(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_setting", message)


class UnknownVoice(DomainError):
    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        super().__init__("unknown_voice", f"unknown voice {name!r}")
        self.name = name
        self.available = available


class PlatformError(DomainError):
    def __init__(self, message: str = "platform not supported yet") -> None:
        super().__init__("unsupported_platform", message)
