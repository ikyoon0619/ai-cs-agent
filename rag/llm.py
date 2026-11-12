from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


load_dotenv()

def get_llm():
    return ChatAnthropic(model='claude-haiku-5-5')

if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("쏘나타를 한 문장으로 소개해 줘")   # 호출하는 메서드
    print(response.content)                                  # 답변 글자가 들어 있는 속성                         # 답변 글자가 들어 있는 속성