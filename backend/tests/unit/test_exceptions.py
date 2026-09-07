from __future__ import annotations

from app.core.exceptions import (
    ErrorCode,
    ImageTooLargeError,
    InferenceError,
    PixelForgeError,
)


def test_problem_document_has_required_rfc9457_members() -> None:
    error = ImageTooLargeError(
        "Your image is larger than the 16 MP limit.",
        technical="input=9000x9000 limit=16000000",
        context={"limitPixels": 16_000_000},
    )

    problem = error.to_problem()

    assert problem["status"] == 413
    assert problem["code"] == ErrorCode.IMAGE_TOO_LARGE.value
    assert problem["detail"] == "Your image is larger than the 16 MP limit."
    assert problem["technical"] == "input=9000x9000 limit=16000000"
    assert problem["context"] == {"limitPixels": 16_000_000}


def test_technical_detail_is_omitted_when_absent() -> None:
    problem = InferenceError("Enhancement failed.").to_problem()

    assert "technical" not in problem
    assert "context" not in problem


def test_every_error_subclass_declares_a_distinct_code() -> None:
    subclasses = PixelForgeError.__subclasses__()

    assert len(subclasses) > 5
    for subclass in subclasses:
        assert isinstance(subclass.code, ErrorCode)
        assert 400 <= subclass.status_code < 600
