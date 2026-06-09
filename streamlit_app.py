"""
강남대 RAG 챗봇 
streamlit run streamlit_app_v2.py
"""

import os
import requests
from bs4 import BeautifulSoup
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ────────────────────────────────────────────────────────────────
# 환경 변수
# ────────────────────────────────────────────────────────────────
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, Exception):
        pass  # 로컬 실행 시 secrets.toml 없어도 무시

# ────────────────────────────────────────────────────────────────
# 페이지 설정
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="강남대 학사 챗봇 - RAG vs LLM 비교", page_icon="🎓", layout="wide")
st.title("🎓 강남대학교 학사 정보 챗봇")

# ────────────────────────────────────────────────────────────────
# 벡터 DB / LLM 로드
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 벡터 DB 로딩 중...")
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

if not os.path.exists("faiss_index"):
    st.error("❌ `faiss_index` 폴더가 없습니다. `python indexer.py` 로 인덱스를 먼저 생성하세요.")
    st.stop()

vectorstore = load_vectorstore()
llm = load_llm()

# ────────────────────────────────────────────────────────────────
# 프롬프트
# ────────────────────────────────────────────────────────────────
system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 친절한 AI 조교입니다. "
    "반드시 아래 제공된 Context만을 기반으로 질문에 답변하세요.\n\n"
    "■ 답변 작성 규칙:\n"
    "1. 질문에서 묻는 핵심 정보만 직접적으로 답하세요.\n"
    "2. 불필요한 배경 설명·반복·일반론은 추가하지 마세요.\n"
    "3. 답변은 3~5문장 이내로 간결하게 작성하세요.\n"
    "4. 출처(조항·페이지)는 답변 끝에 한 줄로만 표기하세요.\n\n"
    "■ 정보가 없을 때 규칙:\n"
    "Context에서 답을 찾을 수 없으면 정확히 다음 한 줄만 출력하세요:\n"
    "[NO_INFO]\n\n"
    "{dept_hint}"
    "Context:\n{context}"
)
web_system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 친절한 AI 조교입니다. "
    "아래는 강남대학교 공식 웹사이트에서 가져온 최신 정보입니다. "
    "이 내용을 바탕으로 질문에 친절하고 정확하게 답변하세요.\n\n"
    "웹사이트 Context:\n{context}"
)

# ★ 순수 LLM용 프롬프트 (RAG 컨텍스트 없이 자체 지식으로만 답변)
pure_llm_system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 AI 조교입니다. "
    "별도의 문서나 데이터베이스 없이, 당신이 사전 학습으로 알고 있는 "
    "일반적인 대학교 학사 규정과 강남대학교에 대한 지식만으로 답변하세요.\n\n"
    "■ 답변 작성 규칙:\n"
    "1. 질문에서 묻는 핵심 정보만 직접적으로 답하세요.\n"
    "2. 확실하지 않은 정보는 '일반적으로', '보통'과 같은 표현을 사용하세요.\n"
    "3. 답변은 3~5문장 이내로 간결하게 작성하세요.\n"
    "4. 강남대학교의 정확한 규정은 공식 홈페이지나 학사팀에 확인을 권장하세요."
)

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
web_prompt = ChatPromptTemplate.from_messages([("system", web_system_prompt), ("human", "{input}")])
pure_llm_prompt = ChatPromptTemplate.from_messages([("system", pure_llm_system_prompt), ("human", "{input}")])

# ────────────────────────────────────────────────────────────────
# 쿼리 확장
# ────────────────────────────────────────────────────────────────
QUERY_SYNONYMS = {
    "휴학": "휴학 휴학원 학적변동 휴학신청 휴학절차 휴학방법 제출 승인",
    "복학": "복학 복학신청 복학절차 학적복귀 복학원",
    "수강신청": "수강신청 강좌신청 수강변경 수강취소 강의등록",
    "졸업": "졸업 졸업요건 졸업학점 졸업심사 학위취득",
    "장학": "장학 장학금 장학생 등록금 감면",
    "성적": "성적 학점 GPA 성적정정 성적이의",
    "결석": "결석 출석 출결 과락 수업시간 결강",
    "전과": "전과 전부 전부전과 전공변경 학과변경 자격",
}

def expand_query(question: str) -> str:
    for keyword, expansion in QUERY_SYNONYMS.items():
        if keyword in question:
            return question + " " + expansion
    return question

# ────────────────────────────────────────────────────────────────
# 웹 크롤링
# ────────────────────────────────────────────────────────────────
KANGNAM_URLS = {
    "학점": ["https://web.kangnam.ac.kr/menu/fd8c126ac0e81458620beb18302bc271.do?encMenuSeq=fcbd4013ab5b5238ed8ca9186a6bfb32"],
    "재수강": ["https://web.kangnam.ac.kr/menu/fd8c126ac0e81458620beb18302bc271.do?encMenuSeq=d2fca573c753f30f9ae5c79dd740bdcd"],
    "수강신청": ["https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?searchMenuSeq=0"],
    "복수전공": ["https://web.kangnam.ac.kr/menu/b2d1211af4999ac7a3ae1e11ad581860.do"],
    "학사일정": ["https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo=1&searchMenuSeq=116&searchType=ttl&searchValue="],
    "휴학": ["https://web.kangnam.ac.kr/menu/12d2ee44cc4e95562f84a01bf953a054.do"],
    "장학": ["https://web.kangnam.ac.kr/menu/062e41fba927c0c76d1c0e929f931016.do"],
}

def select_urls(question: str) -> list[str]:
    for keyword, urls in KANGNAM_URLS.items():
        if keyword in question:
            return urls
    return []

def crawl_web_context(question: str) -> tuple[str, list[str]]:
    urls = select_urls(question)
    collected_texts, source_urls = [], []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/120.0.0.0 Safari/537.36"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
        except Exception:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        rows = soup.select("table tbody tr, .board-list li, .bbs-list tr")
        board_texts = []
        for row in rows[:20]:
            cells = row.find_all(["td", "li"])
            row_text = " | ".join(c.get_text(strip=True) for c in cells if c.get_text(strip=True))
            if row_text:
                board_texts.append(row_text)
        if board_texts:
            collected_texts.append("[ 공지 목록 ]\n" + "\n".join(board_texts))
            source_urls.append(url)
            continue
        main_area = soup.find("main") or soup.find(id="content") or soup.find(class_="content") or soup.body
        if main_area:
            text = main_area.get_text(separator="\n", strip=True)
            if 50 < len(text) < 8000:
                collected_texts.append(text[:3000])
                source_urls.append(url)
    return "\n\n---\n\n".join(collected_texts), source_urls

# ────────────────────────────────────────────────────────────────
# NO_INFO 감지
# ────────────────────────────────────────────────────────────────
NO_INFO_TOKEN = "[NO_INFO]"
NOT_FOUND_PHRASES = [
    "찾을 수 없", "포함되어 있지 않", "포함하고 있지 않",
    "제공된 정보에는", "정보가 없", "확인되지 않",
    "나와 있지 않", "언급되어 있지 않", "명시되어 있지 않",
]

def is_not_found_answer(answer: str) -> bool:
    if NO_INFO_TOKEN in answer:
        return True
    return any(p in answer for p in NOT_FOUND_PHRASES)

# ────────────────────────────────────────────────────────────────
# ★ 순수 LLM 답변 함수 (RAG 없이)
# ────────────────────────────────────────────────────────────────
def answer_pure_llm(question: str) -> str:
    formatted = pure_llm_prompt.format_messages(input=question)
    return llm.invoke(formatted).content

# ────────────────────────────────────────────────────────────────
# RAG 답변 함수
# ────────────────────────────────────────────────────────────────
def answer_question(question: str, dept: str = "") -> dict:
    k = 10
    expanded = expand_query(question)

    if dept:
        all_docs = vectorstore.similarity_search(expanded, k=30)
        filtered = [d for d in all_docs if dept in d.metadata.get("department", "")]
        if len(filtered) < 3:
            filtered += [d for d in all_docs
                         if dept in d.metadata.get("source", "") and d not in filtered]
        docs = filtered[:k] if filtered else all_docs[:k]
        dept_hint = f"[필터: {dept} 관련 규정 우선 적용]\n\n"
    else:
        docs_with_score = vectorstore.similarity_search_with_score(expanded, k=k)
        docs = [d for d, s in docs_with_score if s < 1.8] or [d for d, _ in docs_with_score]
        dept_hint = ""

    context_text = ""
    for doc in docs:
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", None)
        page_info = f" (p.{page + 1})" if page is not None else ""
        context_text += f"[{source}{page_info}]: {doc.page_content}\n\n"

    formatted = prompt.format_messages(context=context_text, input=question, dept_hint=dept_hint)
    answer = llm.invoke(formatted).content

    web_sources, used_web = [], False
    if is_not_found_answer(answer):
        web_context, web_sources = crawl_web_context(question)
        if web_context:
            web_formatted = web_prompt.format_messages(context=web_context, input=question)
            answer = llm.invoke(web_formatted).content
            used_web = True
        else:
            answer = ("죄송합니다. 학칙 PDF와 강남대 공식 웹사이트 모두에서 "
                      "해당 정보를 찾을 수 없습니다. 학사 일정·최신 공지는 "
                      "강남대학교 홈페이지 또는 교학팀에 문의해 주세요.")

    seen, pdf_sources = set(), []
    for doc in docs:
        fname = doc.metadata.get("source", "학칙")
        page = doc.metadata.get("page", None)
        page_num = page + 1 if page is not None else None
        key = f"{fname}:{page_num}"
        if key not in seen:
            seen.add(key)
            pdf_sources.append({"filename": fname, "page": page_num})

    return {
        "answer": answer,
        "pdf_sources": pdf_sources if not used_web else [],
        "web_sources": web_sources,
        "used_web": used_web,
    }

# ────────────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────────────
DEPT_OPTIONS = [
    "", "사회복지학", "경영학", "세무·법행정", "문화콘텐츠",
    "컴퓨터공학", "인공지능융합", "전자·반도체공학",
    "유아교육", "교육학", "스포츠·체육",
]

with st.sidebar:
    st.header("⚙️ 설정")
    dept = st.selectbox(
        "전공 필터 (선택)", DEPT_OPTIONS,
        format_func=lambda x: "전체" if x == "" else x,
    )
    st.markdown("---")

    with st.expander("💡추천 질문"):
        st.markdown(
            "**RAG가 잘 답하는 질문**\n"
            "- 휴학 신청 방법\n"
            "- 전공 필수 과목이 뭐야?\n"
            "- 학사경고는 평점 몇 점 미만이야?\n"
            "- 휴학은 최대 몇 학기까지 가능해?\n"
            "- 졸업 학점은 몇 점이야?\n"
            "- 특정 학과 졸업 요건"
        )
    st.markdown("---")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown("---")
    st.caption("강남대학교 RAG 챗봇 비교 데모\n2025")

# ────────────────────────────────────────────────────────────────
# session_state 초기화
# ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ────────────────────────────────────────────────────────────────
# 추천 질문 버튼 — 대화가 없을 때만 표시
# ────────────────────────────────────────────────────────────────
FAQ_QUESTIONS = [
    ("🎓 졸업 학점", "졸업 학점은 몇 점이야?"),
    ("📅 수강신청 기간", "수강신청 기간은 언제야?"),
    ("🔁 F학점 재수강 규정", "F학점 재수강 규정을 알려줘"),
    ("📝 휴학 신청 방법", "휴학 신청 방법이 어떻게 돼?"),
    ("📊 학점 취득 인정 범위", "학점 취득 인정 범위가 어떻게 돼?"),
    ("📚 복수전공 신청 조건", "복수전공 신청 조건이 뭐야?"),
]

if not st.session_state.messages:
    st.markdown("##### 💡 자주 묻는 질문")
    cols = st.columns(3)
    for idx, (label, question) in enumerate(FAQ_QUESTIONS):
        with cols[idx % 3]:
            if st.button(label, key=f"faq_{idx}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    st.markdown("---")

# ────────────────────────────────────────────────────────────────
# ★ 비교 레이아웃 헤더 (대화가 있을 때만 표시)
# ────────────────────────────────────────────────────────────────
if st.session_state.messages:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #1e3a5f, #2d6a4f);
                        padding: 12px 20px; border-radius: 10px; margin-bottom: 8px;
                        text-align: center;'>
                <span style='color: white; font-size: 16px; font-weight: bold;'>
                    📚 RAG 답변
                </span><br>
                <span style='color: #a8d8a8; font-size: 12px;'>
                    학칙 문서 + 웹사이트 기반
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #4a1c6e, #7b2d8b);
                        padding: 12px 20px; border-radius: 10px; margin-bottom: 8px;
                        text-align: center;'>
                <span style='color: white; font-size: 16px; font-weight: bold;'>
                    🤖 순수 LLM 답변
                </span><br>
                <span style='color: #d8a8e8; font-size: 12px;'>
                    사전 학습 지식만 사용
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ────────────────────────────────────────────────────────────────
# 대화 렌더링 (좌: RAG, 우: 순수 LLM)
# ────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # 질문은 전체 너비로 표시
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        # 답변은 2컬럼 비교 레이아웃
        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True):
                st.markdown("##### 📚 RAG 답변")
                st.markdown(msg.get("rag_content", ""))
                if msg.get("sources"):
                    with st.expander("📎 출처"):
                        for s in msg["sources"]:
                            st.markdown(f"- {s}")

        with col_right:
            with st.container(border=True):
                st.markdown("##### 🤖 순수 LLM 답변")
                st.markdown(msg.get("llm_content", ""))
                st.caption("⚠️ 문서 검색 없이 AI 사전 지식만 사용")

# ────────────────────────────────────────────────────────────────
# 입력 처리 — chat_input 또는 FAQ 버튼
# ────────────────────────────────────────────────────────────────
user_input = st.chat_input("질문을 입력하세요 (예: 휴학 신청은 어떻게 하나요?)")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ★ 2컬럼 동시 답변 생성
    col_left, col_right = st.columns(2)

    rag_result = None
    llm_answer = None

    with col_left:
        with st.container(border=True):
            st.markdown("##### 📚 RAG 답변")
            with st.spinner("문서 검색 중..."):
                rag_result = answer_question(user_input, dept)
            st.markdown(rag_result["answer"])

            sources_display = []
            if rag_result["pdf_sources"]:
                for s in rag_result["pdf_sources"][:5]:
                    page = f" (p.{s['page']})" if s["page"] else ""
                    sources_display.append(f"📄 {s['filename']}{page}")
            if rag_result["web_sources"]:
                for u in rag_result["web_sources"]:
                    sources_display.append(f"🌐 [{u}]({u})")

            if sources_display:
                with st.expander("📎 출처"):
                    for s in sources_display:
                        st.markdown(f"- {s}")

    with col_right:
        with st.container(border=True):
            st.markdown("##### 🤖 순수 LLM 답변")
            with st.spinner("AI 답변 생성 중..."):
                llm_answer = answer_pure_llm(user_input)
            st.markdown(llm_answer)
            st.caption("⚠️ 문서 검색 없이 AI 사전 지식만 사용")

    # session_state에 저장 (role="assistant"로 통합 저장)
    st.session_state.messages.append({
        "role": "assistant",
        "rag_content": rag_result["answer"] if rag_result else "",
        "llm_content": llm_answer or "",
        "sources": sources_display if rag_result else [],
    })

    st.rerun()
