from rag.load import PDF_PATH, load_pdf
from rag.split import split_docs

pages = load_pdf(PDF_PATH)

# ① 글자가 거의 없는 페이지 찾기
empty_pages = []                          # 빈 리스트 (Java: new ArrayList<>())
for page in pages:                        # 29페이지를 하나씩 꺼내서
    text = page.page_content.strip()      # 앞뒤 공백 제거한 본문
    if len(text) < 50:                    # 50자 미만이면
        label = page.metadata["page_label"]   # 카탈로그에 찍힌 쪽 번호
        empty_pages.append(label)         # 리스트에 추가 (Java: list.add())

print("50자 미만 페이지 수:", len(empty_pages))
print("해당 페이지:", empty_pages)

# ② 청크 크기별 비교
for size in [300, 500, 1000]:
    chunks = split_docs(pages, chunk_size=size, chunk_overlap=50)

    total_length = 0                      # 모든 청크 글자 수 합계
    for chunk in chunks:
        total_length = total_length + len(chunk.page_content)

    count = len(chunks)                   # 청크 개수
    average = total_length / count        # 평균 길이

    print("chunk_size =", size, "→ 청크", count, "개, 평균", round(average), "자")



