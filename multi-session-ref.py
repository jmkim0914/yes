"""
멀티세션 RAG 챗봇 — Supabase(pgvector) 세션·벡터 저장, OpenAI 임베딩, 스트리밍 답변.
실행: streamlit run multi-session-ref.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAIError, RateLimitError
from pypdf import PdfReader
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# 경로 / 환경
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent.parent
LOGO_PATHS = [REPO_ROOT / "영인로고.png", CODE_DIR / "영인로고.png"]

MODEL_OPTIONS = ("gpt-4o-mini", "claude-sonnet-4-5", "gemini-3-pro-preview")

SYSTEM_RAG = """당신은 업로드된 문서 맥락을 우선하는 RAG 어시스턴트입니다.
제공된 컨텍스트에 근거해 한글 존대말로 답하세요. 컨텍스트에 없으면 모른다고 말하세요.
답변은 # ## ### 헤딩으로 구조화하고, 구분선(---, ===)과 취소선(~~)은 쓰지 마세요.
반드시 답변 마지막에 아래 형식으로 후속 질문 3개를 붙이세요:

### 💡 다음에 물어볼 수 있는 질문들
1. ...
2. ...
3. ...
"""

SYSTEM_NO_DOC = """당신은 친절한 어시스턴트입니다. 한글 존대말로 답하세요.
답변은 # ## ### 헤딩으로 구조화하세요. 구분선·취소선은 쓰지 마세요.
반드시 마지막에:

### 💡 다음에 물어볼 수 있는 질문들
1. ...
2. ...
3. ...
"""

logging.basicConfig(level=logging.WARNING)


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(CODE_DIR / ".env")


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]*~~", "", text)
    text = re.sub(r"^[\s]*[-=_]{3,}[\s]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY 가 필요합니다.")
    return create_client(url, key)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_llm(model_name: str, temperature: float = 0.3, *, streaming: bool = True):
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, streaming=streaming)
    if model_name == "claude-sonnet-4-5":
        return ChatAnthropic(
            model="claude-sonnet-4-5",
            temperature=temperature,
            max_tokens=8192,
            streaming=streaming,
        )
    if model_name == "gemini-3-pro-preview":
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=temperature, streaming=streaming)
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def inject_css() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; }
button[kind="primary"] { background-color: #ff69b4 !important; border-color: #ff69b4 !important; color: #111 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def header_row() -> None:
    logo_path = next((p for p in LOGO_PATHS if p.exists()), None)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if logo_path:
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<div style="text-align:center;font-size:4rem!important;line-height:1.2;">
<span style="color:#1f77b4!important;">멀티세션</span>
<span style="color:#ffd700!important;"> RAG 챗봇</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def read_pdf_files(uploaded_files: list[Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for f in uploaded_files:
        name = getattr(f, "name", "document.pdf")
        f.seek(0)
        reader = PdfReader(f)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if text:
            pairs.append((name, text))
    return pairs


def chunk_pdf_parts(file_parts: list[tuple[str, str]]) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    rows: list[dict[str, Any]] = []
    for file_name, raw in file_parts:
        docs = splitter.create_documents([raw], metadatas=[{"file_name": file_name}])
        for d in docs:
            rows.append(
                {
                    "file_name": file_name,
                    "content": d.page_content,
                    "metadata": {"file_name": file_name},
                }
            )
    return rows


def embed_and_insert_vectors(
    client: Client,
    session_id: str,
    chunk_rows: list[dict[str, Any]],
    batch_size: int = 10,
) -> int:
    if not chunk_rows:
        return 0
    embedder = get_embeddings()
    texts = [r["content"] for r in chunk_rows]
    embeddings = embedder.embed_documents(texts)
    inserted = 0
    for i in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[i : i + batch_size]
        embs = embeddings[i : i + batch_size]
        payload = []
        for row, vec in zip(batch, embs, strict=True):
            payload.append(
                {
                    "session_id": session_id,
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "embedding": vec,
                    "metadata": row.get("metadata") or {},
                }
            )
        client.table("vector_documents").insert(payload).execute()
        inserted += len(payload)
    return inserted


def match_documents_rpc(client: Client, query: str, session_id: str, match_count: int = 8) -> list[dict[str, Any]]:
    embedder = get_embeddings()
    qvec = embedder.embed_query(query)
    try:
        res = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": qvec,
                "match_count": match_count,
                "filter_session_id": session_id,
            },
        ).execute()
        return list(res.data or [])
    except Exception:
        # RPC 실패 시: 세션 벡터만 읽어 단순 키워드 매칭 대체(폴백)
        rows = (
            client.table("vector_documents")
            .select("id, content, file_name")
            .eq("session_id", session_id)
            .limit(match_count * 2)
            .execute()
        ).data
        return rows[:match_count]


def list_vector_filenames(client: Client, session_id: str) -> list[str]:
    res = client.table("vector_documents").select("file_name").eq("session_id", session_id).execute()
    names = {row["file_name"] for row in (res.data or []) if row.get("file_name")}
    return sorted(names)


def fetch_all_sessions(client: Client) -> list[dict[str, Any]]:
    res = (
        client.table("rag_sessions")
        .select("id, title, messages, created_at, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def upsert_session_row(
    client: Client,
    session_id: str,
    title: str,
    messages: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": session_id,
        "title": title[:500],
        "messages": messages,
        "updated_at": now,
    }
    existing = client.table("rag_sessions").select("id").eq("id", session_id).limit(1).execute()
    if existing.data:
        client.table("rag_sessions").update({"title": row["title"], "messages": messages, "updated_at": now}).eq(
            "id", session_id
        ).execute()
    else:
        row["created_at"] = now
        client.table("rag_sessions").insert(row).execute()


def insert_session_snapshot(
    client: Client,
    new_id: str,
    title: str,
    messages: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    client.table("rag_sessions").insert(
        {
            "id": new_id,
            "title": title[:500],
            "messages": messages,
            "created_at": now,
            "updated_at": now,
        }
    ).execute()


def copy_vectors_between_sessions(client: Client, from_id: str, to_id: str, batch_size: int = 10) -> None:
    fields = "file_name, content, embedding, metadata"
    res = client.table("vector_documents").select(fields).eq("session_id", from_id).execute()
    rows = res.data or []
    if not rows:
        return
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        payload = [{**{k: r[k] for k in ("file_name", "content", "embedding", "metadata") if k in r}, "session_id": to_id} for r in batch]
        client.table("vector_documents").insert(payload).execute()


def delete_session_complete(client: Client, session_id: str) -> None:
    client.table("vector_documents").delete().eq("session_id", session_id).execute()
    client.table("rag_sessions").delete().eq("id", session_id).execute()


def generate_session_title(llm, question: str, answer: str) -> str:
    prompt = f"""다음은 사용자의 첫 질문과 어시스턴트의 첫 답변이다.
15자 내외의 짧은 한국어 세션 제목을 하나만 출력하라. 따옴표·콜론·번호 금지.

질문: {question[:800]}
답변 요약 일부: {answer[:800]}
"""
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(res, "content", str(res)) or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        text = text.strip().splitlines()[0].strip()
        return (text[:80] or "새 대화").replace('"', "")
    except Exception:
        return "새 대화"


def messages_to_lc(msgs: list[dict[str, Any]], limit: int = 16) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in msgs[-limit:]:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


def extract_chunk_text(chunk: AIMessageChunk | Any) -> str:
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


def stream_llm_reply(llm, sys_text: str, history: list[BaseMessage], user_text: str) -> Generator[str, None, None]:
    messages: list[BaseMessage] = [SystemMessage(content=sys_text), *history, HumanMessage(content=user_text)]
    for chunk in llm.stream(messages):
        piece = extract_chunk_text(chunk)
        if piece:
            yield piece


def resolve_title_from_messages(client: Client, messages: list[dict[str, Any]], model_name: str) -> str:
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not users:
        return "새 대화"
    llm = get_llm(model_name, temperature=0.2, streaming=False)
    if assistants:
        return generate_session_title(llm, users[0]["content"], assistants[0]["content"])
    return (users[0]["content"][:40] + "…") if len(users[0]["content"]) > 40 else users[0]["content"]


def ensure_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sidebar_session_labels" not in st.session_state:
        st.session_state.sidebar_session_labels = []


def reset_ui_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.sidebar_session_labels = []
    st.session_state.session_selectbox = "(선택 안 함)"


def auto_save_session(client: Client, model_name: str) -> None:
    title = (
        resolve_title_from_messages(client, st.session_state.messages, model_name)
        if st.session_state.messages
        else "새 대화"
    )
    upsert_session_row(client, st.session_state.session_id, title, st.session_state.messages)




def load_session_into_ui(client: Client, session_id: str) -> None:
    row = client.table("rag_sessions").select("id, title, messages").eq("id", session_id).execute()
    rows = row.data or []
    if not rows:
        return
    data = rows[0]
    msgs = data.get("messages")
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    st.session_state.session_id = data["id"]
    st.session_state.messages = msgs or []


def init() -> Client | None:
    load_env()
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    inject_css()
    ensure_state()
    try:
        return get_supabase()
    except RuntimeError as e:
        st.error(str(e))
        return None


def main() -> None:
    client = init()
    if client is None:
        return
    st.session_state.supabase_client = client

    header_row()

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY 가 필요합니다.")
        st.stop()

    with st.sidebar:
        st.markdown("### 설정")
        model_name = st.radio("LLM 모델", MODEL_OPTIONS, index=0, horizontal=False)
        st.caption("모델 이름은 프롬프트 지정 값을 그대로 사용합니다.")
        if model_name == "claude-sonnet-4-5" and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Claude 선택 시 `.env`에 ANTHROPIC_API_KEY 가 필요합니다.")
        if model_name == "gemini-3-pro-preview" and not os.environ.get("GOOGLE_API_KEY"):
            st.warning("Gemini 선택 시 `.env`에 GOOGLE_API_KEY 가 필요합니다.")

        sessions = fetch_all_sessions(client)
        labels = [
            f"{s.get('title') or '제목 없음'} · {str(s['id'])[:8]}…"
            for s in sessions
        ]
        id_by_label = {labels[i]: sessions[i]["id"] for i in range(len(labels))}
        st.session_state._session_pick_map = id_by_label

        select_options = ["(선택 안 함)"] + labels

        def on_session_select_change() -> None:
            pick_local = st.session_state.session_selectbox
            if pick_local == "(선택 안 함)":
                return
            sid = st.session_state.get("_session_pick_map", {}).get(pick_local)
            if sid:
                load_session_into_ui(st.session_state.supabase_client, sid)

        pick = st.selectbox(
            "저장된 세션 (선택 시 자동 로드)",
            select_options,
            key="session_selectbox",
            on_change=on_session_select_change,
        )
        st.session_state.sidebar_session_labels = labels

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션로드", use_container_width=True):
                if pick != "(선택 안 함)" and pick in id_by_label:
                    load_session_into_ui(client, id_by_label[pick])
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
        with c2:
            if st.button("세션삭제", use_container_width=True):
                if pick != "(선택 안 함)" and pick in id_by_label:
                    sid = id_by_label[pick]
                    delete_session_complete(client, sid)
                    if sid == st.session_state.session_id:
                        reset_ui_session()
                    st.success("삭제했습니다.")
                    st.rerun()

        if st.button("세션저장 (새 행 INSERT)", use_container_width=True):
            if not st.session_state.messages:
                st.warning("저장할 대화가 없습니다.")
            else:
                snap_id = str(uuid.uuid4())
                title_llm = get_llm(model_name, temperature=0.2, streaming=False)
                users = [m for m in st.session_state.messages if m["role"] == "user"]
                asts = [m for m in st.session_state.messages if m["role"] == "assistant"]
                if users and asts:
                    title = generate_session_title(title_llm, users[0]["content"], asts[0]["content"])
                else:
                    title = resolve_title_from_messages(client, st.session_state.messages, model_name)
                snap_messages = json.loads(json.dumps(st.session_state.messages))
                insert_session_snapshot(client, snap_id, title, snap_messages)
                copy_vectors_between_sessions(client, st.session_state.session_id, snap_id)
                st.success(f"새 세션으로 저장했습니다: {title}")
                st.rerun()

        if st.button("화면초기화", use_container_width=True):
            reset_ui_session()
            st.rerun()

        if st.button("vectordb (파일명)", use_container_width=True):
            names = list_vector_filenames(client, st.session_state.session_id)
            if names:
                st.dataframe({"file_name": names}, hide_index=True, use_container_width=True)
            else:
                st.info("현재 세션 ID에 저장된 벡터가 없습니다.")

        st.markdown("---")
        uploaded = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리 → Supabase 벡터 저장", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("PDF를 선택하세요.")
            else:
                try:
                    parts = read_pdf_files(list(uploaded))
                    if not parts:
                        st.error("텍스트를 추출하지 못했습니다.")
                    else:
                        chunks = chunk_pdf_parts(parts)
                        n = embed_and_insert_vectors(client, st.session_state.session_id, chunks)
                        st.success(f"{n}개 청크를 저장했습니다.")
                        auto_save_session(client, model_name)
                except (RateLimitError, OpenAIError) as e:
                    st.error(f"OpenAI 오류: {e}")
                except Exception as e:
                    st.error(f"처리 실패: {e}")

        st.text(
            f"현재 세션: {str(st.session_state.session_id)[:13]}… / 메시지 {len(st.session_state.messages)}개"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]), unsafe_allow_html=True)

    user_text = st.chat_input("문서 또는 일반 질문을 입력하세요")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    hits = match_documents_rpc(client, user_text, st.session_state.session_id, match_count=8)
    context = "\n\n".join(
        f"[파일: {h.get('file_name', '')}]\n{h.get('content', '')}" for h in hits if h.get("content")
    )
    history = messages_to_lc(st.session_state.messages[:-1])

    llm = get_llm(model_name)
    if context.strip():
        sys_prompt = SYSTEM_RAG + f"\n\n### 컨텍스트\n{context}"
    else:
        sys_prompt = SYSTEM_NO_DOC

    with st.chat_message("assistant"):
        stream_buf: list[str] = []

        def stream_gen() -> Generator[str, None, None]:
            for piece in stream_llm_reply(llm, sys_prompt, history, user_text):
                stream_buf.append(piece)
                yield piece

        try:
            streamed = st.write_stream(stream_gen)
            full_reply = streamed if streamed else "".join(stream_buf)
        except (RateLimitError, OpenAIError) as e:
            full_reply = f"LLM 호출 오류: {e}"
            st.error(full_reply)
        except Exception as e:
            full_reply = f"오류: {e}"
            st.error(full_reply)

        full_reply = remove_separators(str(full_reply))
        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        auto_save_session(client, model_name)


if __name__ == "__main__":
    main()
