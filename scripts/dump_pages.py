# scripts/dump_pages.py
# 카탈로그 각 페이지의 본문을 텍스트 파일로 저장 → 평가셋 정답 페이지를 찾을 때 검색용
from rag.load import PDF_PATH, load_pdf

if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    with open("eval/pages.txt", "w", encoding="utf-8") as f:
        for page in pages:
            label = page.metadata["page_label"]          # 카탈로그에 찍힌 쪽 번호
            text = " ".join(page.page_content.split())   # 줄바꿈·여러 칸 공백을 한 칸으로
            f.write(f"===== p.{label} =====\n{text}\n\n")
    print(f"{len(pages)}페이지 → eval/pages.txt 저장")