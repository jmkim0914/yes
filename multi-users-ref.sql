-- multi-users-ref.sql
-- 멀티유저(로그인) + 멀티세션 + 벡터 저장용 Supabase 스키마

DROP FUNCTION IF EXISTS public.match_vector_documents(vector, integer, uuid, uuid);
DROP TABLE IF EXISTS public.vector_documents;
DROP TABLE IF EXISTS public.rag_sessions;

CREATE EXTENSION IF NOT EXISTS vector;

-- 사용자별 세션 저장
CREATE TABLE public.rag_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title text NOT NULL DEFAULT '새 대화',
    messages jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX rag_sessions_owner_updated_idx
    ON public.rag_sessions (owner_user_id, updated_at DESC);

-- 사용자별 벡터 청크 저장
CREATE TABLE public.vector_documents (
    id bigserial PRIMARY KEY,
    owner_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id uuid NOT NULL REFERENCES public.rag_sessions(id) ON DELETE CASCADE,
    file_name text NOT NULL,
    content text NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX vector_documents_owner_session_idx
    ON public.vector_documents (owner_user_id, session_id);

CREATE INDEX vector_documents_embedding_hnsw_idx
    ON public.vector_documents
    USING hnsw (embedding vector_cosine_ops);

-- 사용자/세션 필터 기반 유사도 검색
CREATE OR REPLACE FUNCTION public.match_vector_documents(
    query_embedding vector(1536),
    match_count integer,
    filter_session_id uuid,
    filter_owner_user_id uuid
)
RETURNS TABLE (
    id bigint,
    content text,
    file_name text,
    similarity double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        vd.id,
        vd.content,
        vd.file_name,
        (1 - (vd.embedding <=> query_embedding))::double precision AS similarity
    FROM public.vector_documents vd
    WHERE vd.session_id = filter_session_id
      AND vd.owner_user_id = filter_owner_user_id
    ORDER BY vd.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Row Level Security
ALTER TABLE public.rag_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vector_documents ENABLE ROW LEVEL SECURITY;

-- auth.uid() 기준 본인 데이터만 접근 가능
CREATE POLICY "rag_sessions_owner_select"
    ON public.rag_sessions FOR SELECT
    USING (auth.uid() = owner_user_id);

CREATE POLICY "rag_sessions_owner_insert"
    ON public.rag_sessions FOR INSERT
    WITH CHECK (auth.uid() = owner_user_id);

CREATE POLICY "rag_sessions_owner_update"
    ON public.rag_sessions FOR UPDATE
    USING (auth.uid() = owner_user_id)
    WITH CHECK (auth.uid() = owner_user_id);

CREATE POLICY "rag_sessions_owner_delete"
    ON public.rag_sessions FOR DELETE
    USING (auth.uid() = owner_user_id);

CREATE POLICY "vector_documents_owner_select"
    ON public.vector_documents FOR SELECT
    USING (auth.uid() = owner_user_id);

CREATE POLICY "vector_documents_owner_insert"
    ON public.vector_documents FOR INSERT
    WITH CHECK (auth.uid() = owner_user_id);

CREATE POLICY "vector_documents_owner_update"
    ON public.vector_documents FOR UPDATE
    USING (auth.uid() = owner_user_id)
    WITH CHECK (auth.uid() = owner_user_id);

CREATE POLICY "vector_documents_owner_delete"
    ON public.vector_documents FOR DELETE
    USING (auth.uid() = owner_user_id);
