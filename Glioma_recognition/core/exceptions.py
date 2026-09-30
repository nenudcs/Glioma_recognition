class CompetitionError(Exception):
    """Base class for expected pipeline failures."""


class InvalidRequestError(CompetitionError):
    pass


class InvalidInputError(CompetitionError):
    pass


class MissingSeriesError(CompetitionError):
    pass


class ModelInferenceError(CompetitionError):
    pass


class InvalidTaskResultError(CompetitionError):
    pass


class OutputValidationError(CompetitionError):
    pass

