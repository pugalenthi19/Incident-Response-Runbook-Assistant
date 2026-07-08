"""
AI Incident Response Assistant — Streamlit front end.

Ties together:
- database.py   -> SQLite incident history
- rag.py        -> retrieval-augmented assistant (Chroma + Groq)
- ingest.py     -> (run separately) builds the vector store from data/docs & data/runbook

Run with:  streamlit run app.py
"""

import time
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from pypdf import PdfReader

from database import (
    create_tables,
    get_all_incidents,
    search_incidents,
    insert_feedback,
    get_feedback_stats,
)

# The assistant depends on a built vector store + a valid GROQ_API_KEY in .env.
# Neither is guaranteed to be in place, so the whole app must survive this failing.
try:
    from rag import assistant, MODEL_NAME, EMBEDDING_MODEL, PROMPT
    ASSISTANT_READY = True
    ASSISTANT_ERROR = None
except Exception as e:
    assistant = None
    PROMPT = None
    ASSISTANT_READY = False
    ASSISTANT_ERROR = str(e)
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


st.set_page_config(
    page_title="Incident Response Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

create_tables()

# ============================================================
# Constants
# ============================================================

DOCS_PATH = Path("data/docs")
RUNBOOK_PATH = Path("data/runbook")
VECTOR_DB_PATH = Path("vectorstore")

PAGES = [
    "🏠 Home",
    "🤖 Incident Assistant",
    "📊 Analytics",
    "📖 Runbook Explorer",
    "📚 Documentation Explorer",
    "🏗️ Architecture",
    "ℹ️ About",
]

SEVERITY_BADGES = {
    "Critical": "🔴 Critical",
    "High": "🟠 High",
    "Medium": "🟡 Medium",
    "Low": "🟢 Low",
}

# Mirrors the keyword list in rag.py's IncidentAssistant.search_database,
# so the incident *cards* shown in the UI match what the LLM was told about.
INCIDENT_KEYWORDS = [
    "docker", "kubernetes", "crashloopbackoff", "imagepullbackoff",
    "pod", "nginx", "linux", "aws", "dns", "tcp", "memory", "disk",
    "ec2", "iam", "network", "container",
]

EXAMPLE_QUESTIONS = [
    "How do I fix a CrashLoopBackOff in Kubernetes?",
    "What causes a 502 Bad Gateway in NGINX?",
    "How do I troubleshoot DNS resolution failures?",
    "What should I check when an EC2 instance is unreachable?",
]

# ============================================================
# Session state
# ============================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "settings" not in st.session_state:
    st.session_state.settings = {"top_k": 6, "temperature": 0.0}

# ============================================================
# Light styling polish
# ============================================================

st.markdown(
    """
    <style>
        .stMetric {
            background-color: rgba(151, 166, 195, 0.10);
            padding: 14px 12px;
            border-radius: 10px;
        }
        div[data-testid="stChatMessage"] { padding-top: 4px; padding-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================


def severity_badge(sev):
    if not sev:
        return "⚪ Unknown"
    return SEVERITY_BADGES.get(sev, f"⚪ {sev}")


def get_incident_stats():
    try:
        rows = get_all_incidents()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def count_pdfs(folder: Path):
    try:
        if not folder.exists():
            return 0
        return len(list(folder.rglob("*.pdf")))
    except Exception:
        return 0


def list_pdfs(folder: Path):
    try:
        if not folder.exists():
            return []
        return sorted(folder.rglob("*.pdf"))
    except Exception:
        return []


def vector_db_exists():
    try:
        return VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir())
    except Exception:
        return False


def get_vector_chunk_count():
    if not ASSISTANT_READY or assistant is None:
        return None
    try:
        return assistant.vector_db._collection.count()
    except Exception:
        try:
            data = assistant.vector_db.get()
            return len(data.get("ids", []))
        except Exception:
            return None


def find_similar_incidents(question):
    q = question.lower()
    keyword = next((w for w in INCIDENT_KEYWORDS if w in q), None)
    if not keyword:
        return []
    try:
        return search_incidents(keyword)
    except Exception:
        return []


def classify_sources(docs):
    """Splits retrieved chunks into runbook vs. documentation groups based on
    which data/ folder they were ingested from (real metadata, not a guess),
    so the UI and the exported report can attribute sources by type."""
    runbook_docs, documentation_docs = [], []
    for d in docs:
        source = str(d.metadata.get("source", "")).lower()
        (runbook_docs if "runbook" in source else documentation_docs).append(d)
    return runbook_docs, documentation_docs


def save_feedback(turn, value):
    """Records a thumbs up/down for a turn. Updates in-memory state immediately
    (so the UI reflects it right away) and best-effort persists to SQLite —
    a DB hiccup here should never break the chat experience."""
    turn["feedback"] = value
    try:
        insert_feedback(turn["question"], turn["answer"], value, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass


def ask_with_settings(question, top_k, temperature):
    """Runs the RAG pipeline honoring the sidebar's Top-K / Temperature settings.
    Falls back to the assistant's default ask() if anything about the custom
    path fails, so a settings tweak can never break a question."""
    try:
        retriever = assistant.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 20), "lambda_mult": 0.7},
        )
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        incidents_text = assistant.search_database(question)
        llm = assistant.llm.bind(temperature=temperature)
        chain = PROMPT | llm
        response = chain.invoke({"context": context, "incidents": incidents_text, "question": question})
        return {"answer": response.content, "documents": docs}
    except Exception:
        return assistant.ask(question)


def run_assistant(question):
    start_time = time.time()
    top_k = st.session_state.settings["top_k"]
    temperature = st.session_state.settings["temperature"]
    error = None

    with st.status("Analyzing your question...", expanded=True) as status:
        st.write("🔎 Searching documentation and runbooks...")
        time.sleep(0.2)

        st.write("🗄️ Checking incident history...")
        similar = find_similar_incidents(question)

        st.write("🤖 Generating answer with AI...")
        try:
            result = ask_with_settings(question, top_k, temperature)
        except Exception as e:
            result = {"answer": f"I ran into an error while answering that:\n\n`{e}`", "documents": []}
            error = str(e)

        if error:
            status.update(label="⚠️ Something went wrong", state="error", expanded=True)
        else:
            status.update(label="✅ Answer ready", state="complete", expanded=False)

    elapsed = time.time() - start_time

    return {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": result["answer"],
        "documents": result.get("documents", []),
        "similar_incidents": similar,
        "elapsed": elapsed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error,
        "feedback": None,
    }


def build_text_report():
    lines = [
        "INCIDENT RESPONSE ASSISTANT — CONVERSATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    for i, turn in enumerate(st.session_state.conversation, start=1):
        lines.append(f"[{i}] QUESTION: {turn['question']}")
        lines.append(f"    {turn['timestamp']}  ({turn['elapsed']:.2f}s)")
        lines.append("")
        lines.append("ANSWER:")
        lines.append(turn["answer"])
        lines.append("")

        runbook_docs, documentation_docs = classify_sources(turn.get("documents") or [])
        runbook_names = sorted({Path(str(d.metadata.get("source", "Unknown"))).name for d in runbook_docs})
        doc_names = sorted({Path(str(d.metadata.get("source", "Unknown"))).name for d in documentation_docs})
        similar = turn.get("similar_incidents") or []

        if runbook_names or doc_names or similar:
            lines.append("SOURCES USED:")
            for n in runbook_names:
                lines.append(f"  [Runbook] {n}")
            for n in doc_names:
                lines.append(f"  [Documentation] {n}")
            for row in similar:
                lines.append(f"  [Similar Incident] #{row['id']} {row['title']} [{row['severity']}] -> {row['resolution']}")
            lines.append("")

        if turn.get("feedback"):
            lines.append(f"FEEDBACK: {'Helpful' if turn['feedback'] == 'helpful' else 'Not Helpful'}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("")
    return "\n".join(lines)


def render_incident_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([5, 2])
        c1.markdown(f"**#{row['id']} — {row['title']}**")
        c2.markdown(severity_badge(row["severity"]))
        st.caption(
            f"{row['category'] or 'Uncategorized'} · {row['affected_service'] or '—'} · "
            f"Team: {row['assigned_team'] or '—'} · Status: {row['status'] or '—'}"
        )
        rc_col, res_col = st.columns(2)
        with rc_col:
            st.markdown("**Root Cause**")
            st.write(row["root_cause"] or "—")
        with res_col:
            st.markdown("**Resolution**")
            st.write(row["resolution"] or "—")


def render_source_chunk(doc):
    source_name = Path(str(doc.metadata.get("source", "Unknown"))).name
    page = doc.metadata.get("page")
    page_label = f" — Page {page + 1}" if isinstance(page, int) else ""
    st.markdown(f"**📄 {source_name}**{page_label}")
    preview = (doc.page_content or "").strip()
    if len(preview) > 400:
        preview = preview[:400] + "..."
    st.text(preview if preview else "(empty chunk)")
    st.divider()


def render_answer_content(turn):
    if turn.get("error"):
        st.error(turn["answer"])
    else:
        st.markdown(turn["answer"])

    docs = turn.get("documents") or []
    runbook_docs, documentation_docs = classify_sources(docs)
    similar = turn.get("similar_incidents") or []

    if similar:
        with st.expander(f"📋 Similar Incidents ({len(similar)})"):
            for row in similar:
                render_incident_card(row)

    if runbook_docs:
        with st.expander(f"📖 Retrieved Runbooks ({len(runbook_docs)})"):
            for d in runbook_docs:
                render_source_chunk(d)

    if documentation_docs:
        with st.expander(f"📚 Retrieved Documentation ({len(documentation_docs)})"):
            for d in documentation_docs:
                render_source_chunk(d)

    runbook_names = sorted({Path(str(d.metadata.get("source", "Unknown"))).name for d in runbook_docs})
    doc_names = sorted({Path(str(d.metadata.get("source", "Unknown"))).name for d in documentation_docs})
    total_sources = len(runbook_names) + len(doc_names) + len(similar)

    if total_sources:
        with st.expander(f"🔗 Sources Used ({total_sources})"):
            if runbook_names:
                st.markdown("**📖 Runbook**")
                for n in runbook_names:
                    st.write(n)
            if doc_names:
                st.markdown("**📄 Documentation**")
                for n in doc_names:
                    st.write(n)
            if similar:
                st.markdown("**🗄️ Similar Incident**")
                for row in similar:
                    st.write(f"Incident #{row['id']} — {row['title']}")

    st.caption(f"⏱️ {turn['elapsed']:.2f}s · {turn['timestamp']}")

    if not turn.get("error"):
        render_feedback_widget(turn)


def render_feedback_widget(turn):
    if turn.get("feedback"):
        icon = "👍" if turn["feedback"] == "helpful" else "👎"
        st.caption(f"{icon} Thanks for your feedback!")
        return

    st.caption("Was this answer helpful?")
    fb_col1, fb_col2, _ = st.columns([1.3, 1.6, 3])
    with fb_col1:
        if st.button("👍 Helpful", key=f"fb_up_{turn['id']}", use_container_width=True):
            save_feedback(turn, "helpful")
            st.rerun()
    with fb_col2:
        if st.button("👎 Not Helpful", key=f"fb_down_{turn['id']}", use_container_width=True):
            save_feedback(turn, "not_helpful")
            st.rerun()


# ============================================================
# Pages
# ============================================================


def render_home_page():
    st.title("🛠️ AI Incident Response Assistant")
    st.markdown(
        "A retrieval-augmented assistant that helps IT teams diagnose and resolve "
        "incidents using official documentation, internal runbooks, and historical "
        "incident data."
    )
    st.divider()

    incidents_df = get_incident_stats()
    total_chunks = get_vector_chunk_count()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Incidents", len(incidents_df))
    c2.metric("📖 Runbooks", count_pdfs(RUNBOOK_PATH))
    c3.metric("📚 Documentation", count_pdfs(DOCS_PATH))
    c4.metric("🧩 Vector Chunks", total_chunks if total_chunks is not None else "N/A")

    st.divider()

    c5, c6, c7 = st.columns(3)
    c5.metric("🤖 LLM", MODEL_NAME)
    c6.metric("🔤 Embedding Model", EMBEDDING_MODEL.split("/")[-1])
    c7.metric("🗄️ Database", "SQLite")

    st.divider()

    st.subheader("System Status")
    s1, s2, s3 = st.columns(3)
    s1.write("🟢 **Database** — Connected")
    s2.write(("🟢" if vector_db_exists() else "🟡") + f" **Vector Store** — {'Ready' if vector_db_exists() else 'Not built yet'}")
    s3.write(("🟢" if ASSISTANT_READY else "🔴") + f" **AI Assistant** — {'Online' if ASSISTANT_READY else 'Offline'}")

    if not ASSISTANT_READY:
        st.warning(
            "The AI assistant couldn't start up. Run `python ingest.py` to build the vector "
            f"database and make sure `GROQ_API_KEY` is set in your `.env` file.\n\nDetails: `{ASSISTANT_ERROR}`"
        )

    st.divider()

    if st.button("🚀 Start Assistant", type="primary", use_container_width=True):
        st.session_state.page = "🤖 Incident Assistant"
        st.rerun()


def render_assistant_page():
    st.title("🤖 Incident Assistant")
    st.caption("Ask about an incident and get AI-powered guidance from documentation, runbooks, and incident history.")

    if not ASSISTANT_READY:
        st.error("The assistant isn't available right now.")
        st.info(
            "This usually means the vector database hasn't been built yet, or `GROQ_API_KEY` "
            "is missing from your `.env` file.\n\n"
            "1. Add PDFs to `data/docs/` and `data/runbook/`\n"
            "2. Run `python ingest.py` to build the vector database\n"
            "3. Add `GROQ_API_KEY=your_key_here` to a `.env` file in the project root\n"
            "4. Restart the app"
        )
        with st.expander("Technical details"):
            st.code(str(ASSISTANT_ERROR))
        return

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.session_state.conversation:
            st.download_button(
                "📄 Export Report (TXT)",
                data=build_text_report(),
                file_name=f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
    with action_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()

    if not st.session_state.conversation:
        st.markdown("**Try one of these:**")
        ex_cols = st.columns(2)
        for idx, eq in enumerate(EXAMPLE_QUESTIONS):
            if ex_cols[idx % 2].button(eq, key=f"example_{idx}", use_container_width=True):
                st.session_state.pending_question = eq
        st.divider()

    for turn in st.session_state.conversation:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer_content(turn)

    typed_question = st.chat_input("Describe the issue or ask a question...")
    question = typed_question or st.session_state.pending_question
    st.session_state.pending_question = None

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            turn = run_assistant(question)
            render_answer_content(turn)
        st.session_state.conversation.append(turn)
        st.session_state.search_history.append({
            "question": question,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })


def render_analytics_page():
    st.title("📊 Analytics Dashboard")
    st.caption("Insights across all logged incidents.")

    df = get_incident_stats()
    if df.empty:
        st.info("No incident data yet. Run `python seed_database.py` to populate sample data.")
        render_feedback_summary()
        return

    with st.expander("🔍 Filters"):
        f1, f2 = st.columns(2)
        severity_filter = f1.multiselect("Severity", sorted(df["severity"].dropna().unique().tolist()))
        category_filter = f2.multiselect("Category", sorted(df["category"].dropna().unique().tolist()))

    filtered = df.copy()
    if severity_filter:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]

    if filtered.empty:
        st.warning("No incidents match the selected filters.")
        render_feedback_summary()
        return

    st.subheader("Overview")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Total", len(filtered))
    m2.metric("Resolved", int((filtered["status"] == "Resolved").sum()))
    m3.metric("Closed", int((filtered["status"] == "Closed").sum()))
    m4.metric("🔴 Critical", int((filtered["severity"] == "Critical").sum()))
    m5.metric("🟠 High", int((filtered["severity"] == "High").sum()))
    m6.metric("🟡 Medium", int((filtered["severity"] == "Medium").sum()))
    m7.metric("🟢 Low", int((filtered["severity"] == "Low").sum()))

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Incidents by Category")
        st.bar_chart(filtered["category"].dropna().value_counts())
    with chart_col2:
        st.subheader("Incidents by Severity")
        st.bar_chart(filtered["severity"].dropna().value_counts())

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.subheader("Team Workload")
        st.bar_chart(filtered["assigned_team"].dropna().value_counts())
    with chart_col4:
        st.subheader("Service Distribution")
        st.bar_chart(filtered["affected_service"].dropna().value_counts())

    st.subheader("Incident Trend Over Time")
    trend = filtered.copy()
    trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
    trend = trend.dropna(subset=["created_at"])
    if not trend.empty:
        monthly = trend.groupby(trend["created_at"].dt.to_period("M")).size()
        monthly.index = monthly.index.astype(str)
        st.line_chart(monthly)
    else:
        st.caption("No valid dates to chart.")

    render_feedback_summary()


def render_feedback_summary():
    try:
        stats = get_feedback_stats()
    except Exception:
        stats = []

    counts = {row["feedback"]: row["count"] for row in stats}
    helpful = counts.get("helpful", 0)
    not_helpful = counts.get("not_helpful", 0)
    total = helpful + not_helpful

    if total == 0:
        return

    st.divider()
    st.subheader("🤖 Assistant Feedback")
    st.caption("Ratings collected from the 👍 / 👎 buttons on the Incident Assistant page.")
    f1, f2, f3 = st.columns(3)
    f1.metric("👍 Helpful", helpful)
    f2.metric("👎 Not Helpful", not_helpful)
    f3.metric("Helpful Rate", f"{helpful / total * 100:.0f}%")


def render_pdf_explorer(folder: Path, label: str, icon: str):
    st.title(f"{icon} {label}")

    files = list_pdfs(folder)
    if not files:
        st.info(f"No PDF files found in `{folder}/`. Add PDFs there and run `python ingest.py` to index them.")
        return

    search = st.text_input(f"Search {label.lower()}", key=f"search_{label}", placeholder="Filter by filename...")
    filtered_files = [f for f in files if search.lower() in f.name.lower()] if search else files

    if not filtered_files:
        st.warning("No files match your search.")
        return

    selected_name = st.selectbox("Select a document", [f.name for f in filtered_files], key=f"select_{label}")
    selected_path = next(f for f in filtered_files if f.name == selected_name)

    num_pages = None
    try:
        reader = PdfReader(str(selected_path))
        num_pages = len(reader.pages)
    except Exception:
        pass

    info_col, download_col = st.columns([3, 1])
    with info_col:
        st.subheader(selected_path.name)
        details = []
        if num_pages:
            details.append(f"{num_pages} pages")
        try:
            details.append(f"{selected_path.stat().st_size / 1024:.1f} KB")
        except Exception:
            pass
        if details:
            st.caption(" · ".join(details))
    with download_col:
        try:
            with open(selected_path, "rb") as f:
                st.download_button(
                    "⬇️ Download", f.read(), file_name=selected_path.name,
                    mime="application/pdf", use_container_width=True,
                )
        except Exception:
            st.caption("File unavailable")

    with st.expander("📄 Text Preview", expanded=True):
        try:
            reader = PdfReader(str(selected_path))
            preview_text = ""
            for i, page in enumerate(reader.pages):
                if i >= 3:
                    break
                preview_text += (page.extract_text() or "") + "\n"
            preview_text = preview_text.strip()
            if preview_text:
                if len(preview_text) > 3000:
                    preview_text = preview_text[:3000] + "..."
                st.text(preview_text)
            else:
                st.caption("No extractable text found on the first pages (this may be a scanned/image-based PDF).")
        except Exception as e:
            st.caption(f"Couldn't generate a preview for this file. ({e})")


def render_runbook_page():
    render_pdf_explorer(RUNBOOK_PATH, "Runbooks", "📖")


def render_documentation_page():
    render_pdf_explorer(DOCS_PATH, "Documentation", "📚")


def render_architecture_page():
    st.title("🏗️ Architecture")

    dot_source = """
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica", color="#888888"];

        User [fillcolor="#DCEBFF" label="User"];
        Streamlit [fillcolor="#DCEBFF" label="Streamlit UI\\n(app.py)"];
        RAG [fillcolor="#FFE8CC" label="RAG Engine\\n(rag.py)"];
        Chroma [fillcolor="#D9F2D9" label="ChromaDB\\n(vector store)"];
        SQLite [fillcolor="#D9F2D9" label="SQLite\\n(incident history)"];
        Groq [fillcolor="#F5D9F0" label="Groq LLM\\n(Llama 3.3 70B)"];
        Answer [fillcolor="#DCEBFF" label="Answer"];

        User -> Streamlit -> RAG;
        RAG -> Chroma [label="semantic search"];
        RAG -> SQLite [label="keyword search"];
        Chroma -> RAG;
        SQLite -> RAG;
        RAG -> Groq [label="context + question"];
        Groq -> Answer;
    }
    """
    try:
        st.graphviz_chart(dot_source)
    except Exception:
        st.code(
            "User -> Streamlit UI -> RAG Engine\n"
            "                        |--> ChromaDB  (semantic search over docs & runbooks)\n"
            "                        |--> SQLite    (keyword search over incident history)\n"
            "                        `--> Groq LLM  (Llama 3.3 70B) -> Answer",
            language=None,
        )

    st.divider()
    st.subheader("Tech Stack")

    stack_cols = st.columns(3)
    with stack_cols[0]:
        st.markdown("**Frontend**")
        st.markdown("- Streamlit")
    with stack_cols[1]:
        st.markdown("**RAG / AI**")
        st.markdown("- LangChain\n- ChromaDB\n- HuggingFace Embeddings\n- Groq (Llama 3.3 70B)")
    with stack_cols[2]:
        st.markdown("**Data**")
        st.markdown("- SQLite\n- PyPDF")


def render_about_page():
    st.title("ℹ️ About")

    st.markdown(
        "### AI Incident Response Assistant\n\n"
        "A Retrieval-Augmented Generation (RAG) system that helps IT teams diagnose and "
        "resolve incidents faster by combining official documentation, internal runbooks, "
        "and historical incident data with a large language model."
    )
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Developer")
        st.write("**Pugalenthi E**")
        st.write("B.Tech, Electronics & Communication Engineering")
    with c2:
        st.subheader("Project Objective")
        st.write(
            "Demonstrate a working, end-to-end RAG application combining Python, SQL, "
            "LangChain, vector search, and LLM integration."
        )

    st.divider()
    st.subheader("Technologies Used")
    st.write("Python · Streamlit · LangChain · ChromaDB · HuggingFace Embeddings · Groq (Llama 3.3) · SQLite · PyPDF")

    st.divider()
    st.subheader("GitHub Repository")
    st.write("https://github.com/pugalenthi19/Incident-Response-Runbook-Assistant")


# ============================================================
# Sidebar & routing
# ============================================================


# Split into two calls around the page dispatch on purpose: the nav radio has
# to render *before* the page content so a click feels instant, but Recent
# Searches / Settings have to render *after* it — otherwise they'd always
# display state from one interaction ago (e.g. "No searches yet" right after
# you ask your first question, since that question is only recorded while
# the page content below renders).


def render_sidebar_nav():
    with st.sidebar:
        st.markdown("## 🛠️ Incident Assistant")
        st.caption("AI-powered IT incident response")
        st.divider()

        st.session_state.page = st.radio(
            "Navigation", PAGES,
            index=PAGES.index(st.session_state.page),
            label_visibility="collapsed",
        )


def render_sidebar_status():
    with st.sidebar:
        st.divider()
        st.markdown("**System Status**")
        st.write("🟢 Database Connected")
        st.write("🟢 Assistant Ready" if ASSISTANT_READY else "🔴 Assistant Unavailable")
        st.write("🟢 Vector Store Ready" if vector_db_exists() else "🟡 Vector Store Empty")

        st.divider()
        with st.expander("🕓 Recent Searches"):
            if st.session_state.search_history:
                recent = list(reversed(st.session_state.search_history[-10:]))
                for idx, item in enumerate(recent):
                    label = item["question"] if len(item["question"]) <= 40 else item["question"][:37] + "..."
                    if st.button(label, key=f"hist_{idx}", use_container_width=True):
                        st.session_state.pending_question = item["question"]
                        st.session_state.page = "🤖 Incident Assistant"
                        st.rerun()
            else:
                st.caption("No searches yet.")

        with st.expander("⚙️ Settings"):
            st.session_state.settings["top_k"] = st.slider(
                "Top-K Retrieval", 1, 15, st.session_state.settings["top_k"]
            )
            st.session_state.settings["temperature"] = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.settings["temperature"], 0.1
            )
            st.text_input("Model", value=MODEL_NAME, disabled=True)
            st.caption("Top-K and temperature apply live. Model is configured in `rag.py`.")


render_sidebar_nav()

PAGE_RENDERERS = {
    "🏠 Home": render_home_page,
    "🤖 Incident Assistant": render_assistant_page,
    "📊 Analytics": render_analytics_page,
    "📖 Runbook Explorer": render_runbook_page,
    "📚 Documentation Explorer": render_documentation_page,
    "🏗️ Architecture": render_architecture_page,
    "ℹ️ About": render_about_page,
}

PAGE_RENDERERS[st.session_state.page]()

render_sidebar_status()