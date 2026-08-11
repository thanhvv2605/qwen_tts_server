from pydantic import BaseModel, Field, field_validator, model_validator

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
    instruct: str | None = None
    voice_id: str | None = None

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
    def instruct_not_empty(cls, v: "str | None") -> "str | None":
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("instruct must not be empty when provided")
        return v

    @field_validator("voice_id")
    @classmethod
    def voice_id_not_empty(cls, v: "str | None") -> "str | None":
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("voice_id must not be empty when provided")
        return v

    @field_validator("language")
    @classmethod
    def language_supported(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
        return v

    @model_validator(mode="after")
    def exactly_one_of_instruct_or_voice_id(self) -> "TTSRequest":
        if (self.instruct is None) == (self.voice_id is None):
            raise ValueError("exactly one of 'instruct' or 'voice_id' must be provided")
        return self


class JobSubmitRequest(BaseModel):
    items: list[TTSRequest] = Field(min_length=1)
