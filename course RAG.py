import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Any


# ==============================
# Dependency checks
# ==============================
try:
    import streamlit as st
except ModuleNotFoundError:
    print(
        "Missing dependency: streamlit\n"
        "Please install it first with:\n"
        "    pip install streamlit\n\n"
        "If you are using requirements.txt, run:\n"
        "    pip install -r requirements.txt\n\n"
        "Then start the app with:\n"
        "    streamlit run app.py"
    )
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as exc:
    print(
        f"Missing dependency: {exc.name}\n"
        "Please install project dependencies with:\n"
        "    pip install -r requirements.txt"
    )
    sys.exit(1)


# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Course Material RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stTextInput>div>div>input {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Course Material Assistant")
st.subheader("Smart Q&A for Your Course PDFs")
st.caption(
    "Ask questions based on built-in course materials or an uploaded PDF. "
    "Answers are generated only from retrieved document chunks."
)


# ==============================
# App constants
# ==============================
DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-R1"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DOCS_DIR = Path("data")
DEFAULT_DOC_NAME = "course_material.pdf"
DEFAULT_DOC_PATH = DEFAULT_DOCS_DIR / DEFAULT_DOC_NAME

SYSTEM_PROMPT = """
You are a helpful course material assistant.
Answer the user's question only based on the retrieved document excerpts.
Do not invent facts.
If the answer is not explicitly supported by the excerpts, say:
"I could not find that information in the provided course materials."

When possible:
1. Give a concise answer first.
2. Then provide 2-4 bullet points if the question needs explanation.
3. Keep the answer study-friendly and easy to review.
""".strip()

COURSE_KEYWORDS = [
    "definition",
    "concept",
    "example",
    "formula",
    "theorem",
    "hypothesis",
    "assumption",
    "method",
    "result",
    "summary",
    "chapter",
    "lecture",
    "week",
    "regression",
    "matrix",
    "probability",
    "statistics",
]


# ==============================
# Validation helpers
# ==============================
def validate_runtime_inputs(api_key: str, question: str) -> Tuple[bool, str]:
    if not api_key.strip():
        return False, "Please enter your API key in the sidebar."
    if not question.strip():
        return False, "Please enter a question."
    return True, ""


def filter_documents(split_docs: List[Any], keywords: List[str]) -> List[Any]:
    filtered_docs = []
    for doc in split_docs:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in keywords):
            filtered_docs.append(doc)
    return filtered_docs if filtered_docs else split_docs


def run_self_checks() -> None:
    class DummyDoc:
        def __init__(self, page_content: str):
            self.page_content = page_content

    ok, msg = validate_runtime_inputs("abc", "what is regression?")
    assert ok is True and msg == ""

    ok, msg = validate_runtime_inputs("", "what is regression?")
    assert ok is False and "API key" in msg

    ok, msg = validate_runtime_inputs("abc", "   ")
    assert ok is False and "question" in msg

    docs = [DummyDoc("This lecture explains regression assumptions."), DummyDoc("Random unrelated text")]
    filtered = filter_documents(docs, COURSE_KEYWORDS)
    assert len(filtered) == 1
    assert "regression" in filtered[0].page_content.lower()

    docs_no_match = [DummyDoc("hello world"), DummyDoc("another plain sentence")]
    filtered_no_match = filter_documents(docs_no_match, COURSE_KEYWORDS)
    assert len(filtered_no_match) == 2


if __name__ == "__main__" and "streamlit" not in sys.argv[0].lower():
    try:
        run_self_checks()
        print(
            "Self-checks passed.\n"
            "To run the app UI, install dependencies and execute:\n"
            "    streamlit run app.py"
        )
    except AssertionError:
        print("Self-checks failed. Please review the helper functions.")
        sys.exit(1)


# ==============================
# Cached resources
# ==============================
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(docs)
    filtered_docs = filter_documents(split_docs, COURSE_KEYWORDS)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(filtered_docs, embeddings)
    return vectorstore, len(docs), len(split_docs), len(filtered_docs), file_name


# ==============================
# LLM helpers
# ==============================
def build_llm(api_key: str, model_name: str, base_url: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )


def answer_question(llm: ChatOpenAI, retriever, question: str):
    related_docs = retriever.invoke(question)
    context = "\n\n".join(
        [
            f"[Source {idx}]\n{doc.page_content}"
            for idx, doc in enumerate(related_docs, start=1)
        ]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Retrieved document excerpts:
{context}

User question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content, related_docs


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Settings")

    default_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    api_key = st.text_input("SiliconFlow API Key", value=default_api_key, type="password")

    model_name = st.text_input("Model name", value=DEFAULT_MODEL)
    base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)

    top_k = st.slider("Number of retrieved chunks", min_value=2, max_value=8, value=4)

    st.markdown("### Document source")
    uploaded_file = st.file_uploader("Optional: upload your own PDF", type=["pdf"])

    st.markdown("### Notes")
    st.markdown("- If no file is uploaded, the built-in course PDF will be used.")
    st.markdown("- Put your default PDF at `data/course_material.pdf`.")
    st.markdown("- Best for 1-5 course documents combined into one PDF.")


# ==============================
# Load document
# ==============================
file_bytes = None
active_file_name = None

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    active_file_name = uploaded_file.name
else:
    if not DEFAULT_DOC_PATH.exists():
        st.error(
            "Default PDF not found. Please either upload a PDF in the sidebar or place a file at `data/course_material.pdf`."
        )
        st.stop()

    with open(DEFAULT_DOC_PATH, "rb") as f:
        file_bytes = f.read()
    active_file_name = DEFAULT_DOC_NAME


# ==============================
# Build retriever
# ==============================
try:
    vectorstore, page_count, chunk_count, filtered_count, stored_name = build_vectorstore(
        file_bytes,
        active_file_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", page_count)
    col2.metric("Chunks", chunk_count)
    col3.metric("Indexed chunks", filtered_count)

    st.success(f"Document indexed successfully: {stored_name}")
except Exception as exc:
    st.error(f"Failed to process the PDF: {exc}")
    st.stop()


# ==============================
# Main UI
# ==============================
st.markdown("### 💡 Quick Actions")
colA, colB, colC = st.columns(3)

if colA.button("📖 Summarize Document"):
    st.session_state.quick_q = "Summarize the main points of this document"
if colB.button("🧠 Key Concepts"):
    st.session_state.quick_q = "List the key concepts in this material"
if colC.button("📝 Exam Focus"):
    st.session_state.quick_q = "What are the important points for exam preparation"

example_questions = [
    "What are the main topics covered in this material?",
    "Summarize this chapter in simple terms.",
    "What definition does the material give for regression?",
    "What assumptions are mentioned in this document?",
    "Explain this concept like I am a beginner.",
]

selected_example = st.selectbox("Example questions", options=[""] + example_questions)
default_question = selected_example if selected_example else "What are the key points of this material?"

default_val = st.session_state.get("quick_q", default_question)
question = st.text_input("💬 Ask a question", value=default_val)

if st.button("Get answer", type="primary"):
    valid, error_message = validate_runtime_inputs(api_key, question)
    if not valid:
        st.error(error_message)
    else:
        try:
            llm = build_llm(
                api_key=api_key.strip(),
                model_name=model_name.strip(),
                base_url=base_url.strip(),
            )
            with st.spinner("Generating answer..."):
                answer, sources = answer_question(llm, retriever, question.strip())

            st.markdown("### ✅ Answer")
            st.write(answer)

            st.markdown("### 📚 Sources")
            for idx, doc in enumerate(sources, start=1):
                page_num = doc.metadata.get("page")
                page_label = f"Page {page_num + 1}" if isinstance(page_num, int) else "Unknown page"
                with st.expander(f"Source {idx} — {page_label}"):
                    st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Request failed: {exc}")



