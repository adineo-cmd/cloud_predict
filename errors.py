# errors.py
# Custom exception hierarchy for structured, catchable error handling across the pipeline.


class CloudClassifierError(Exception):
    """Base class for all project-specific errors."""
    exit_code = 1


class ModelNotFoundError(CloudClassifierError):
    """Raised when a trained model artifact is missing."""
    exit_code = 2

    def __init__(self, path):
        super().__init__(f"Model not found at '{path}'. Run `python -m cli.train` first.")


class DatasetNotFoundError(CloudClassifierError):
    """Raised when the dataset directory or expected class folders are missing."""
    exit_code = 3


class InvalidImageError(CloudClassifierError):
    """Raised when an input image cannot be read or decoded."""
    exit_code = 4


class ConfigurationError(CloudClassifierError):
    """Raised when config values are inconsistent (e.g. bad threshold, unknown class)."""
    exit_code = 5
