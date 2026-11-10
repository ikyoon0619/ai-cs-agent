"""step 5. 검색한 청크를 근거로 Claude가 구조화된 답(Answer)을 만든다."""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="사용자에게 보여줄 답변")
    sources: list[int] = Field(description="근거가 된 PDF 페이지 번호 (1부터 시작)")
    confidence: float = Field(ge=0, le=1, description="근거가 충분한 정도")


def answer_question(question: str) -> Answer:
    # TODO:
    #   1) rag.retrieve.search로 청크를 찾는다
    #   2) 각 청크 앞에 [p.12]처럼 페이지 번호를 붙여 context 문자열을 만든다
    #   3) settings.prompt_path의 프롬프트에 question, context를 채운다
    #   4) ChatAnthropic(model=settings.llm_model).with_structured_output(Answer)로 호출한다
    raise NotImplementedError


if __name__ == "__main__":
    print(answer_question("트렁크 용량은 얼마야?").model_dump_json(indent=2))
