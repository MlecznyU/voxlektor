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


class ServiceUnavailable(DomainError):
    """The daemon cannot be reached (not running, not answering, or not starting)."""

    def __init__(self, message: str = "voxlektor is not running") -> None:
        super().__init__("service_unavailable", message)


class NotRunning(ServiceUnavailable):
    """Nobody listens on the daemon's socket path (it is missing, refused or not a socket).

    Only this case means "the daemon is off"; a timeout or a broken reply is a daemon that is
    alive but not answering, which stays a plain ``ServiceUnavailable``.
    """

    def __init__(self, message: str = "voxlektor is not running") -> None:
        super().__init__(message)


class PlatformError(DomainError):
    def __init__(self, message: str = "platform not supported yet") -> None:
        super().__init__("unsupported_platform", message)
