"""Custom exceptions for VTON processing pipeline."""


class DataLoadingError(Exception):
    """Custom exception for data loading operations.

    Args:
        message: The error message describing the issue.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"DataLoadingError: {self.message}"


class VTONProcessingError(Exception):
    """Custom exception for VTON processing errors.

    Args:
        message: The error message describing the issue.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"VTONProcessingError: {self.message}"


class MissingDataError(Exception):
    """Custom exception for missing data issues.

    Args:
        message: The error message describing the missing data issue.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"MissingDataError: {self.message}"