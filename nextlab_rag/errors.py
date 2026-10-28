class RagError(RuntimeError):
    """Base error for index and answer operations."""


class IndexNotReadyError(RagError):
    """The local index is absent, incomplete, or does not match the export."""


class RagConfigError(RagError):
    """A required local or chat endpoint setting is missing or invalid."""
