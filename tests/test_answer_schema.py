import pytest
from pydantic import ValidationError

from rag.answer import Answer
from rag.config import settings


def test_answer_accepts_valid_output():
    a = Answer(answer="트렁크는 580L입니다", sources=[12], confidence=0.9)
    assert a.sources == [12]


def test_answer_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Answer(answer="x", sources=[], confidence=1.5)


def test_settings_defaults():
    assert settings.embedding_model == "voyage-3.5"
    assert settings.chunk_size > settings.chunk_overlap
