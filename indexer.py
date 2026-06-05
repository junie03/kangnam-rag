import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── 학과 키워드 매핑 ────────────────────────────────────────────────
# 각 학과/전공에 해당하는 키워드 목록
# PDF 본문에 해당 키워드가 등장하면 그 페이지에 학과 메타데이터를 부여
DEPT_KEYWORD_MAP = {
    # 복지융합대학
    "사회복지학": [
        "사회복지학부", "사회복지학전공", "사회복지사", "사회복지실천",
        "사회서비스학전공", "시니어비즈니스학과", "사회복지"
    ],

    # 경영관리대학
    "경영학": [
        "경영학전공", "상경학부", "경영관리대학", "경영학개론",
        "마케팅관리", "재무관리", "조직행동론", "인적자원관리",
        "국제무역학전공", "경제금융학전공"
    ],
    "세무·법행정": [
        "세무학전공", "법행정학전공", "세무학과", "법학전공",
        "세법", "조세법"
    ],

    # 글로벌문화콘텐츠대학
    "문화콘텐츠": [
        "문화콘텐츠학과", "국제지역학과", "중국콘텐츠비즈니스학과",
        "한국어문학학과", "한국어교육", "문화콘텐츠"
    ],

    # 공과대학
    "컴퓨터공학": [
        "컴퓨터공학부", "소프트웨어전공", "메타버스게임전공",
        "소프트웨어개론", "C프로그래밍", "파이썬응용",
        "데이터사이언스학과", "데이터사이언스", "데이터분석",
        "머신러닝"
    ],
    "인공지능융합": [
        "인공지능융합부", "인공지능전공", "AI융합", "인공지능융합학부",
        "인공지능", "AI", "딥러닝", "자연어처리", "컴퓨터비전"
    ],
    "전자·반도체공학": [
        "전자반도체공학부", "전자공학전공", "반도체공학전공",
        "전기전자공학", "반도체소자", "전자회로", "제어공학", "반도체"
    ],

    # 사범대학
    "유아교육": [
        "유아교육학과", "유아교육", "보육", "유아발달",
        "유아교육과정", "유치원"
    ],
    "교육학": [
        "교육학과", "교육학개론", "교육심리", "교육철학",
        "교육행정", "교육과정연구", "사범대학", "교직"
    ],
    "스포츠·체육": [
        "체육학부", "스포츠", "체육학", "스포츠지도",
        "스포츠경영", "수상스키", "스포츠직무"
    ],
}

# 한 페이지에 여러 학과 키워드가 있을 때 우선순위 (앞쪽 학과 우선)
DEPT_PRIORITY = list(DEPT_KEYWORD_MAP.keys())


def detect_department(text: str) -> str:
    """
    페이지 본문(text)에서 학과 키워드를 탐색해 학과명을 반환.
    매칭이 없으면 빈 문자열 반환.
    매칭된 학과가 여러 개면 키워드 등장 횟수가 가장 많은 학과 반환.
    """
    scores: dict[str, int] = {}
    for dept, keywords in DEPT_KEYWORD_MAP.items():
        count = sum(text.count(kw) for kw in keywords)
        if count > 0:
            scores[dept] = count

    if not scores:
        return ""

    # 점수가 같으면 DEPT_PRIORITY 순서로 결정
    best = max(scores, key=lambda d: (scores[d], -DEPT_PRIORITY.index(d)))
    return best


def build_vector_db():
    data_dir = "data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"❌ {data_dir} 폴더가 없어 생성했습니다. PDF 파일들을 넣고 다시 실행하세요.")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"❌ {data_dir} 폴더에 PDF 파일이 없습니다.")
        return

    print(f"발견된 PDF 파일: {pdf_files}")

    all_pages = []

    # 1. 모든 PDF 파일 로드 + 학과 메타데이터 부여
    for pdf in pdf_files:
        pdf_path = os.path.join(data_dir, pdf)
        print(f"--- [{pdf}] 로딩 중... ---")
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            dept_counter: dict[str, int] = {}

            for page in pages:
                page.metadata["source"] = pdf

                # ★ 본문 내용으로 학과 자동 감지
                detected = detect_department(page.page_content)
                page.metadata["department"] = detected

                if detected:
                    dept_counter[detected] = dept_counter.get(detected, 0) + 1

            all_pages.extend(pages)

            # 파일별 학과 분포 출력
            if dept_counter:
                print(f"  📚 학과 분포: {dept_counter}")
            else:
                print(f"  ℹ️  학과 미감지 (공통 규정 PDF)")

        except Exception as e:
            print(f"❌ {pdf} 로드 중 에러 발생: {e}")

    if not all_pages:
        print("로드된 페이지가 없습니다.")
        return

    print(f"\n2. 전체 {len(all_pages)}페이지를 청크로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_pages)

    print(f"   -> 생성된 총 문서 조각: {len(docs)}개")

    # 청크 단위 학과 분포 통계
    chunk_dept_counter: dict[str, int] = {}
    for doc in docs:
        d = doc.metadata.get("department", "")
        label = d if d else "(미분류)"
        chunk_dept_counter[label] = chunk_dept_counter.get(label, 0) + 1

    print(f"\n   [청크 학과 분포]")
    for dept, cnt in sorted(chunk_dept_counter.items(), key=lambda x: -x[1]):
        print(f"    {dept}: {cnt}개")

    print("\n3. 통합 벡터 DB 생성 중 (OpenAI Embeddings)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("\n✅ 모든 PDF 데이터가 'faiss_index'로 통합 저장되었습니다!")


if __name__ == "__main__":
    build_vector_db()
