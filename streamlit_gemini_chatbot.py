# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from google import genai  # For interacting with the Google Gemini API

# Optional/extra imports (gracefully handled below if missing)
try:
    import io
    from pypdf import PdfReader
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    _PDF_FEATURES_AVAILABLE = True
except Exception:
    _PDF_FEATURES_AVAILABLE = False

# --- 1. Page Configuration and Title ---

st.set_page_config(page_title="Gemini Chatbot", page_icon="💬")
st.title("💬 Gemini Chatbot")
st.caption("A simple and friendly chat using Google's Gemini Flash model — now with PDF Q&A & Download Chat")

# --- 2. Sidebar for Settings ---
with st.sidebar:
    st.subheader("Settings")

    # API key input
    google_api_key = st.text_input("Google AI API Key", type="password")

    # Reset conversation button
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

    st.markdown("---")
    st.subheader("Documents")

    # If optional deps not installed, show a gentle hint
    if not _PDF_FEATURES_AVAILABLE:
        st.info(
            "PDF Q&A needs extra packages. Install: `pip install pypdf scikit-learn`",
            icon="ℹ️",
        )

    # File uploader (works even if deps missing; we'll guard when processing)
    uploaded_files = st.file_uploader(
        "Upload PDF (maks 3 file)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Dokumen ini akan diindeks lokal untuk membantu jawaban.",
    )

    use_docs = st.checkbox("Gunakan dokumen saat menjawab", value=True, help="Jika aktif, jawaban akan mengacu pada potongan paling relevan dari PDF yang diunggah.")

    # Download chat history feature
    st.markdown("---")
    st.subheader("Export")
    if "messages" in st.session_state and st.session_state.get("messages"):
        chat_text = "\n\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            label="📥 Download Chat History (.txt)",
            data=io.BytesIO(chat_text.encode("utf-8")),
            file_name="chat_history.txt",
            mime="text/plain",
            disabled=False,
        )
    else:
        st.caption("Tidak ada percakapan untuk diunduh.")

# --- 3. API Key and Client Initialization ---
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# Create Gemini client only when needed or API key changed
if ("genai_client" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        st.session_state.genai_client = genai.Client(api_key=google_api_key)
        st.session_state._last_key = google_api_key
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        # also clear doc index on key change just to be safe
        st.session_state.pop("doc_chunks", None)
        st.session_state.pop("doc_vectorizer", None)
        st.session_state.pop("doc_matrix", None)
        st.session_state.pop("doc_files", None)
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        st.stop()

# --- PDF Utilities (Mini-RAG lokal) ---
# Only define when optional deps are available
if _PDF_FEATURES_AVAILABLE:
    def extract_text_from_pdf(file) -> str:
        """Ekstrak semua teks dari satu file PDF (streamlit UploadedFile)."""
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                # some PDFs might error on specific pages; continue
                continue
        return "\n".join(texts)

    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
        """Potong teks panjang menjadi potongan beririsan agar tetap koheren."""
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    def build_tfidf_index(chunks):
        """Bangun index TF-IDF dari daftar chunk."""
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1,
            stop_words=None,
        )
        matrix = vectorizer.fit_transform(chunks)
        return vectorizer, matrix

    def retrieve_chunks(query: str, vectorizer, matrix, chunks, k: int = 5):
        """Ambil k chunk paling relevan untuk query berdasarkan cosine similarity."""
        if not query.strip():
            return []
        q_vec = vectorizer.transform([query])
        sims = linear_kernel(q_vec, matrix).ravel()
        top_idx = sims.argsort()[::-1][:k]
        results = [(chunks[i], float(sims[i])) for i in top_idx]
        return results

    def format_context_snippet(results):
        """Gabungkan potongan konteks dengan skor ringan agar transparan."""
        if not results:
            return "(no relevant context found)"
        lines = []
        for i, (chunk, score) in enumerate(results, 1):
            lines.append(f"[{i}] (score={score:.3f})\n{chunk}")
        return "\n\n---\n\n".join(lines)

# --- 4. Build/refresh document index when files uploaded ---
if uploaded_files and _PDF_FEATURES_AVAILABLE:
    all_text = []
    file_names = []
    for f in uploaded_files[:3]:
        try:
            text = extract_text_from_pdf(f)
            if text and text.strip():
                all_text.append(text)
                file_names.append(f.name)
            else:
                st.sidebar.warning(f"Tidak ada teks yang bisa diambil dari: {f.name}")
        except Exception as e:
            st.sidebar.warning(f"Gagal memproses {f.name}: {e}")

    if all_text:
        concat = "\n\n".join(all_text)
        chunks = chunk_text(concat, chunk_size=800, overlap=120)
        if chunks:
            try:
                vec, mat = build_tfidf_index(chunks)
                st.session_state.doc_chunks = chunks
                st.session_state.doc_vectorizer = vec
                st.session_state.doc_matrix = mat
                st.session_state.doc_files = file_names
                st.sidebar.success(f"✅ Terindeks {len(chunks)} potongan dari {len(file_names)} file.")
            except Exception as e:
                st.sidebar.error(f"Gagal membangun indeks: {e}")
        else:
            st.sidebar.info("Tidak ada chunk teks yang valid dari PDF.")

# Show the indexed files list (if any)
if st.session_state.get("doc_files"):
    with st.sidebar:
        st.caption("📚 Indexed files:")
        for fn in st.session_state.doc_files:
            st.write(f"- {fn}")

# --- 5. Chat History Management ---
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.genai_client.chats.create(model="gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle reset
if reset_button:
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("doc_chunks", None)
    st.session_state.pop("doc_vectorizer", None)
    st.session_state.pop("doc_matrix", None)
    st.session_state.pop("doc_files", None)
    st.rerun()

# --- 6. Display Past Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. Handle User Input and API Communication ---
prompt = st.chat_input("Type your message here...")

if prompt:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build final prompt (optionally augmented with document context)
    try:
        final_prompt = prompt

        if (
            use_docs
            and _PDF_FEATURES_AVAILABLE
            and all(k in st.session_state for k in ["doc_chunks", "doc_vectorizer", "doc_matrix"])
        ):
            top = retrieve_chunks(
                query=prompt,
                vectorizer=st.session_state.doc_vectorizer,
                matrix=st.session_state.doc_matrix,
                chunks=st.session_state.doc_chunks,
                k=5,
            )
            context_block = format_context_snippet(top)
            system_hint = (
                "You are a helpful assistant that must answer using ONLY the provided context when relevant. "
                "If the answer is not found in the context, say that you don't have enough information and suggest next steps."
            )
            final_prompt = (
                f"{system_hint}\n\n"
                f"### Context from uploaded documents:\n{context_block}\n\n"
                f"### User question:\n{prompt}\n\n"
                f"### Instructions:\n- Cite which snippet numbers you used (e.g., [1], [3])."
            )

        # Send to Gemini
        response = st.session_state.chat.send_message(final_prompt)

        if hasattr(response, "text"):
            answer = response.text
        else:
            answer = str(response)

    except Exception as e:
        answer = f"An error occurred: {e}"

    # Display assistant message and save
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
