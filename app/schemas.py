from pydantic import BaseModel, Field, field_validator

SUPPORTED_LANGUAGES = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}


class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    instruct: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 2000:
            raise ValueError("text must be at most 2000 characters")
        return v

    @field_validator("instruct")
    @classmethod
    def instruct_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("instruct must not be empty")
        return v

    @field_validator("language")
    @classmethod
    def language_supported(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
        return v


class JobSubmitRequest(BaseModel):
    items: list[TTSRequest] = Field(min_length=1)
