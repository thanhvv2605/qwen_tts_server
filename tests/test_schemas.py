import pytest
from pydantic import ValidationError

from app.schemas import TTSRequest, JobSubmitRequest


def test_valid_request():
    req = TTSRequest(text="Hello", language="English", instruct="calm voice")
    assert req.text == "Hello"
    assert req.language == "English"
    assert req.instruct == "calm voice"


def test_default_language_is_auto():
    req = TTSRequest(text="Hello", instruct="calm voice")
    assert req.language == "Auto"


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="   ", language="English", instruct="calm voice")


def test_text_too_long_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="x" * 2001, language="English", instruct="calm voice")


def test_empty_instruct_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", language="English", instruct="  ")


def test_unsupported_language_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", language="Klingon", instruct="calm voice")


def test_job_submit_request_valid():
    req = JobSubmitRequest(
        items=[
            {"text": "hello", "language": "English", "instruct": "calm voice"},
            {"text": "world", "instruct": "excited voice"},
        ]
    )
    assert len(req.items) == 2
    assert req.items[0].text == "hello"
    assert req.items[1].language == "Auto"


def test_job_submit_request_rejects_empty_items():
    with pytest.raises(ValidationError):
        JobSubmitRequest(items=[])


def test_job_submit_request_rejects_invalid_item():
    with pytest.raises(ValidationError):
        JobSubmitRequest(items=[{"text": "", "language": "English", "instruct": "calm"}])
