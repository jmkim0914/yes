"""
멀티유저 + 멀티세션 RAG 챗봇 (Supabase Auth + pgvector)
실행: streamlit run multi-users-ref.py
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAIError, RateLimitError
from pypdf import PdfReader
from supabase import Client, create_client

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


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]*~~", "", text)
    text = re.sub(r"^[\s]*[-=_]{3,}[\s]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY)가 필요합니다.")
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
<div style="text-align:center;font-size:2.5rem!important;line-height:1.2;">
<span style="color:#1f77b4!important;">PDF 기반</span>
<span style="color:#ffd700!important;"> 멀티유저 멀티세션</span><br/>
<span style="color:#ff69b4!important;">RAG 챗봇</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def ensure_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None


def reset_ui_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.session_selectbox = "(선택 안 함)"


def set_runtime_api_keys() -> None:
    st.markdown("### API Key 설정")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", key="openai_api_key_input")
    anthropic_key = st.text_input("ANTHROPIC_API_KEY", type="password", key="anthropic_api_key_input")
    gemini_key = st.text_input("GOOGLE_API_KEY", type="password", key="gemini_api_key_input")

    # Secrets(os.getenv) 우선, 없으면 사이드바 입력값으로 런타임 주입
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key.strip()
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key.strip()


def parse_auth_user(resp: Any) -> dict[str, Any] | None:
    user = getattr(resp, "user", None)
    if user is None and getattr(resp, "session", None) is not None:
        user = getattr(resp.session, "user", None)
    if user is None and isinstance(resp, dict):
        user = resp.get("user") or (resp.get("session") or {}).get("user")

    if user is None:
        return None

    uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)
    if not uid:
        return None
    return {"id": uid, "email": email or ""}


def auth_panel(client: Client) -> bool:
    st.markdown("### 로그인 / 회원가입")
    if st.session_state.auth_user:
        st.success(f"로그인됨: {st.session_state.auth_user.get('email') or st.session_state.auth_user['id']}")
        if st.button("로그아웃", use_container_width=True):
            try:
                client.auth.sign_out()
            except Exception:
                pass
            st.session_state.auth_user = None
            reset_ui_session()
            st.rerun()
        return True

    mode = st.radio("인증 모드", ("로그인", "회원가입"), horizontal=True)
    email = st.text_input("Login ID (이메일)", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")
    if st.button(mode, type="primary", use_container_width=True):
        if not email.strip() or not password.strip():
            st.warning("이메일과 비밀번호를 입력하세요.")
        else:
            try:
                if mode == "로그인":
                    resp = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                else:
                    resp = client.auth.sign_up({"email": email.strip(), "password": password})
                user_info = parse_auth_user(resp)
                if user_info:
                    st.session_state.auth_user = user_info
                    reset_ui_session()
                    st.success(f"{mode} 성공")
                    st.rerun()
                else:
                    st.info("회원가입 후 이메일 인증이 필요할 수 있습니다. 인증 후 다시 로그인해 주세요.")
            except Exception as e:
                st.error(f"{mode} 실패: {e}")
    return False


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
            rows.append({"file_name": file_name, "content": d.page_content, "metadata": {"file_name": file_name}})
    return rows


def embed_and_insert_vectors(
    client: Client,
    user_id: str,
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
                    "owner_user_id": user_id,
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


def match_documents_rpc(client: Client, query: str, user_id: str, session_id: str, match_count: int = 8) -> list[dict[str, Any]]:
    embedder = get_embeddings()
    qvec = embedder.embed_query(query)
    try:
        res = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": qvec,
                "match_count": match_count,
                "filter_session_id": session_id,
                "filter_owner_user_id": user_id,
            },
        ).execute()
        return list(res.data or [])
    except Exception:
        rows = (
            client.table("vector_documents")
            .select("id, content, file_name")
            .eq("owner_user_id", user_id)
            .eq("session_id", session_id)
            .limit(match_count * 2)
            .execute()
        ).data
        return rows[:match_count]


def fetch_all_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    res = (
        client.table("rag_sessions")
        .select("id, title, messages, created_at, updated_at")
        .eq("owner_user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def upsert_session_row(
    client: Client,
    user_id: str,
    session_id: str,
    title: str,
    messages: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    existing = client.table("rag_sessions").select("id").eq("owner_user_id", user_id).eq("id", session_id).limit(1).execute()
    if existing.data:
        client.table("rag_sessions").update({"title": title[:500], "messages": messages, "updated_at": now}).eq(
            "owner_user_id", user_id
        ).eq("id", session_id).execute()
    else:
        client.table("rag_sessions").insert(
            {
                "owner_user_id": user_id,
                "id": session_id,
                "title": title[:500],
                "messages": messages,
                "created_at": now,
                "updated_at": now,
            }
        ).execute()


def insert_session_snapshot(
    client: Client,
    user_id: str,
    new_id: str,
    title: str,
    messages: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    client.table("rag_sessions").insert(
        {
            "owner_user_id": user_id,
            "id": new_id,
            "title": title[:500],
            "messages": messages,
            "created_at": now,
            "updated_at": now,
        }
    ).execute()


def copy_vectors_between_sessions(client: Client, user_id: str, from_id: str, to_id: str, batch_size: int = 10) -> None:
    fields = "file_name, content, embedding, metadata"
    res = (
        client.table("vector_documents")
        .select(fields)
        .eq("owner_user_id", user_id)
        .eq("session_id", from_id)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        payload = []
        for r in batch:
            payload.append(
                {
                    "owner_user_id": user_id,
                    "session_id": to_id,
                    "file_name": r["file_name"],
                    "content": r["content"],
                    "embedding": r["embedding"],
                    "metadata": r.get("metadata") or {},
                }
            )
        client.table("vector_documents").insert(payload).execute()


def delete_session_complete(client: Client, user_id: str, session_id: str) -> None:
    client.table("vector_documents").delete().eq("owner_user_id", user_id).eq("session_id", session_id).execute()
    client.table("rag_sessions").delete().eq("owner_user_id", user_id).eq("id", session_id).execute()


def list_vector_filenames(client: Client, user_id: str, session_id: str) -> list[str]:
    res = (
        client.table("vector_documents")
        .select("file_name")
        .eq("owner_user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    names = {row["file_name"] for row in (res.data or []) if row.get("file_name")}
    return sorted(names)


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


def resolve_title_from_messages(messages: list[dict[str, Any]], model_name: str) -> str:
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not users:
        return "새 대화"
    llm = get_llm(model_name, temperature=0.2, streaming=False)
    if assistants:
        return generate_session_title(llm, users[0]["content"], assistants[0]["content"])
    return (users[0]["content"][:40] + "…") if len(users[0]["content"]) > 40 else users[0]["content"]


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


def auto_save_session(client: Client, user_id: str, model_name: str) -> None:
    title = resolve_title_from_messages(st.session_state.messages, model_name) if st.session_state.messages else "새 대화"
    upsert_session_row(client, user_id, st.session_state.session_id, title, st.session_state.messages)


def load_session_into_ui(client: Client, user_id: str, session_id: str) -> None:
    row = (
        client.table("rag_sessions")
        .select("id, title, messages")
        .eq("owner_user_id", user_id)
        .eq("id", session_id)
        .execute()
    )
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
    st.set_page_config(page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
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
    header_row()

    with st.sidebar:
        set_runtime_api_keys()
        st.markdown("---")
        logged_in = auth_panel(client)
        if not logged_in:
            st.info("로그인 후 챗봇 기능을 사용할 수 있습니다.")
            return
        user_id = st.session_state.auth_user["id"]

        st.markdown("### 설정")
        model_name = st.radio("LLM 모델", MODEL_OPTIONS, index=0, horizontal=False)
        if model_name == "gpt-4o-mini" and not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI 모델 사용 시 OPENAI_API_KEY가 필요합니다.")
        if model_name == "claude-sonnet-4-5" and not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("Claude 모델 사용 시 ANTHROPIC_API_KEY가 필요합니다.")
        if model_name == "gemini-3-pro-preview" and not os.getenv("GOOGLE_API_KEY"):
            st.warning("Gemini 모델 사용 시 GOOGLE_API_KEY가 필요합니다.")

        sessions = fetch_all_sessions(client, user_id)
        labels = [f"{s.get('title') or '제목 없음'} · {str(s['id'])[:8]}…" for s in sessions]
        id_by_label = {labels[i]: sessions[i]["id"] for i in range(len(labels))}
        select_options = ["(선택 안 함)"] + labels

        def on_session_select_change() -> None:
            pick_local = st.session_state.session_selectbox
            if pick_local == "(선택 안 함)":
                return
            sid = id_by_label.get(pick_local)
            if sid:
                load_session_into_ui(client, user_id, sid)

        pick = st.selectbox(
            "저장된 세션 (선택 시 자동 로드)",
            select_options,
            key="session_selectbox",
            on_change=on_session_select_change,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션로드", use_container_width=True):
                if pick != "(선택 안 함)" and pick in id_by_label:
                    load_session_into_ui(client, user_id, id_by_label[pick])
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
        with c2:
            if st.button("세션삭제", use_container_width=True):
                if pick != "(선택 안 함)" and pick in id_by_label:
                    sid = id_by_label[pick]
                    delete_session_complete(client, user_id, sid)
                    if sid == st.session_state.session_id:
                        reset_ui_session()
                    st.success("삭제했습니다.")
                    st.rerun()

        if st.button("세션저장 (새 행 INSERT)", use_container_width=True):
            if not st.session_state.messages:
                st.warning("저장할 대화가 없습니다.")
            else:
                snap_id = str(uuid.uuid4())
                title = resolve_title_from_messages(st.session_state.messages, model_name)
                snap_messages = json.loads(json.dumps(st.session_state.messages))
                insert_session_snapshot(client, user_id, snap_id, title, snap_messages)
                copy_vectors_between_sessions(client, user_id, st.session_state.session_id, snap_id)
                st.success(f"새 세션으로 저장했습니다: {title}")
                st.rerun()

        if st.button("화면초기화", use_container_width=True):
            reset_ui_session()
            st.rerun()

        if st.button("vectordb (파일명)", use_container_width=True):
            names = list_vector_filenames(client, user_id, st.session_state.session_id)
            if names:
                st.dataframe({"file_name": names}, hide_index=True, use_container_width=True)
            else:
                st.info("현재 세션 ID에 저장된 벡터가 없습니다.")

        st.markdown("---")
        uploaded = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리 → Supabase 벡터 저장", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("임베딩 생성을 위해 OPENAI_API_KEY를 입력해 주세요.")
            elif not uploaded:
                st.warning("PDF를 선택하세요.")
            else:
                try:
                    parts = read_pdf_files(list(uploaded))
                    if not parts:
                        st.error("텍스트를 추출하지 못했습니다.")
                    else:
                        chunks = chunk_pdf_parts(parts)
                        n = embed_and_insert_vectors(client, user_id, st.session_state.session_id, chunks)
                        st.success(f"{n}개 청크를 저장했습니다.")
                        auto_save_session(client, user_id, model_name)
                except (RateLimitError, OpenAIError) as e:
                    st.error(f"OpenAI 오류: {e}")
                except Exception as e:
                    st.error(f"처리 실패: {e}")

        st.text(f"현재 세션: {str(st.session_state.session_id)[:13]}… / 메시지 {len(st.session_state.messages)}개")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]), unsafe_allow_html=True)

    user_text = st.chat_input("문서 또는 일반 질문을 입력하세요")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    user_id = st.session_state.auth_user["id"]
    hits = match_documents_rpc(client, user_text, user_id, st.session_state.session_id, match_count=8)
    context = "\n\n".join(f"[파일: {h.get('file_name', '')}]\n{h.get('content', '')}" for h in hits if h.get("content"))
    history = messages_to_lc(st.session_state.messages[:-1])

    llm = get_llm(model_name)
    sys_prompt = SYSTEM_RAG + f"\n\n### 컨텍스트\n{context}" if context.strip() else SYSTEM_NO_DOC

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
        auto_save_session(client, user_id, model_name)


if __name__ == "__main__":
    main()
