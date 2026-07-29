"""Domain exceptions for the runtime."""


class EngineError(Exception):
    """Raised when the inference engine encounters an error."""


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found on HuggingFace Hub."""


class ExpertExtractionError(Exception):
    """Raised when expert weights cannot be extracted from a checkpoint."""
