from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[list[Recommendation]] = None
    end_of_conversation: bool = False


class CatalogItem(BaseModel):
    entity_id: str
    name: str
    link: str
    description: str = ""
    keys: list[str] = Field(default_factory=list)
    test_type: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    searchable_text: str = ""


class ExtractedSlots(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    domain: Optional[str] = None
    skills: Optional[str] = None
    language_pref: Optional[str] = None
    assessment_types: Optional[str] = None
    constraints: Optional[str] = None
    purpose: Optional[str] = None

    def signal_count(self) -> int:
        return sum([
            bool(self.role),
            bool(self.seniority),
            bool(self.domain),
            bool(self.skills),
            bool(self.assessment_types),
            bool(self.purpose),
        ])


class RouterAction(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"