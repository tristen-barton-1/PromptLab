import os
import io
import re
import time
import json
import zipfile
import html, csv
from io import BytesIO
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
import pdfplumber
from docx import Document


try:
    import streamlit as st
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Non-Technical Prompt Lab", page_icon="🧪", layout="wide")

st.title("🧪 Non-Technical Environment for Developing, Testing, and Refining Assistant Prompts")
st.caption("Write/refine **system instructions** and **prompts**, upload a **document**, and test responses with **document context** — no code required.")

# ================= Sidebar: API & Settings =================
with st.sidebar:
    st.header("Settings")

    backend = st.selectbox(
        "Backend",
        ["Completions", "Assistant"],
        index=0,
        help=(
            "• **Completions**: Fast, single-turn chat. Great for quick tests.\n"
            "• **Assistants (beta)**: Uses the Assistants API run/threads flow. "
            "Choose this to mirror production behavior."
        )
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"],
        index=0,
        help="Smaller/mini models respond faster; larger models may be more detailed."
    )

    max_output_tokens = st.slider(
        "Max output tokens",
        128, 2048, 512, 64,
        help="Upper limit on the length of the AI's answer. Higher = potentially longer responses."
    )
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.2, 0.1,
        help="Lower values make answers more consistent; higher values make them more creative."
    )

    st.divider()
    st.write("**Retrieval Settings**")

    chunk_chars = st.slider(
        "Chunk size",
        1000, 6000, 3000, 500,
        help=(
            "We split your uploaded document into slices called *chunks*. "
            "This sets how big each slice is (in characters). Larger chunks include more context "
            "but can be slower and less targeted. Tip: 2,000–4,000 works well for most RFPs."
        )
    )

    top_k = st.slider(
        "Top-K chunks",
        1, 8, 3,
        help=(
            "For each prompt, we pick the **Top-K** most relevant slices of the document to send to the AI. "
            "Higher values add more context (useful for broad questions) but can slow things down or dilute relevance. "
            "Tip: start at 3; raise it if answers seem to miss details."
        )
    )

    st.caption("The app automatically selects the Top-K most relevant chunks per prompt based on simple keyword matching.")

    st.divider()
    st.write("**Export Options**")
    export_format = st.multiselect(
        "Choose export format(s)",
        ["CSV", "JSON"],
        default=["CSV", "JSON"],
        help="Pick the file types you want to download."
    )

# ================= Section 1: Instructions & Prompts =================
st.markdown("## 1) Instructions & Prompts")
st.markdown("Describe how the assistant should behave, then list the prompts you want to test.")

system_instructions = st.text_area(
    "System Instructions",
    height=120,
    placeholder="e.g., You are an assistant that helps draft RFP responses. Prioritize compliance and clarity...",
    key="system_instructions",
)

default_prompts = pd.DataFrame([
    {"prompt": "Summarize the key requirements in this RFP.", "expected_result": ""},
    {"prompt": "List the compliance items and indicate any gaps.", "expected_result": ""},
])
prompts_df = st.data_editor(
    st.session_state.get("prompts_df", default_prompts),
    num_rows="dynamic",
    use_container_width=True,
    key="prompts_editor"
)
st.session_state["prompts_df"] = prompts_df

col1, col2 = st.columns([1,1])
with col1:
    if st.button("💾 Download Prompt Set (JSON)"):
        cfg = {
            "system_instructions": system_instructions,
            "prompts": prompts_df.to_dict(orient="records"),
        }
        st.download_button(
            "Download prompts.json",
            data=json.dumps(cfg, indent=2).encode("utf-8"),
            file_name="prompts.json",
            mime="application/json",
        )
with col2:
    uploaded_cfg = st.file_uploader("⬆️ Load Prompt Set (JSON)", type=["json"], key="cfg_uploader")
    if uploaded_cfg:
        try:
            cfg = json.loads(uploaded_cfg.read().decode("utf-8"))
            system_instructions = cfg.get("system_instructions", system_instructions)
            prompts_df = pd.DataFrame(cfg.get("prompts", [])) if cfg.get("prompts") else prompts_df
            st.session_state["prompts_df"] = prompts_df
            st.success("Prompt set loaded.")
        except Exception as e:
            st.error(f"Failed to load prompt set: {e}")

# ================= Section 2: Document Upload =================
st.markdown("## 2) Upload Document (PDF, DOCX, or TXT)")
uploaded = st.file_uploader("Drop a file or click to upload", type=["pdf", "docx", "txt"], key="doc_uploader")

def parse_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)

def parse_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

doc_text, doc_name = "", None
if uploaded:
    doc_name = uploaded.name
    bytes_data = uploaded.read()
    try:
        if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
            with st.spinner("Parsing PDF..."):
                doc_text = parse_pdf(bytes_data)
        elif uploaded.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"] or uploaded.name.lower().endswith(".docx"):
            with st.spinner("Parsing DOCX..."):
                doc_text = parse_docx(bytes_data)
        elif uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
            with st.spinner("Reading TXT..."):
                doc_text = parse_txt(bytes_data)
        else:
            st.error("Unsupported file type.")
        st.success(f"Document parsed: {doc_name} ({len(doc_text)} characters)")
    except Exception as e:
        st.error(f"Parsing failed: {e}")

# ================= Helpers =================
def chunk_text(text: str, max_chars: int) -> list[str]:
    """Simple fixed-size chunking by characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

_word_re = re.compile(r"[A-Za-z0-9']+")

def _tokenize(s: str) -> list[str]:
    return [w for w in _word_re.findall((s or "").lower()) if len(w) > 2]

def select_top_k_chunks(chunks: list[str], query: str, k: int = 3) -> list[str]:
    """Very lightweight relevance scoring based on word overlap with the query."""
    if not chunks:
        return []
    qset = set(_tokenize(query))
    if not qset:
        return chunks[:k]
    scored = []
    for idx, ch in enumerate(chunks):
        toks = _tokenize(ch)
        overlap = len(set(toks) & qset)
        scored.append((overlap, idx, ch))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    top = [ch for _, _, ch in scored[:k]]
    if all(score == 0 for score, _, _ in scored[:k]):
        return chunks[:k]
    return top

def call_openai_completions(messages):
    """Chat Completions backend."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed in this environment.")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_output_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def call_openai_assistants(system_instructions: str, context: str, prompt: str):
    """Assistants (beta) backend — minimal flow: create assistant, thread, run, poll, read reply."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed in this environment.")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    assistant = client.beta.assistants.create(
        name="BidWERX Prompt Lab",
        instructions=system_instructions or "",
        model=model,
    )

    thread = client.beta.threads.create()
    user_content = f"Use this document context:\n\n{context}\n\nUser prompt:\n{prompt}"
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    for _ in range(60):
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "completed":
            break
        if run.status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Assistant run ended with status: {run.status}")
        time.sleep(1)

    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    if msgs.data:
        content_parts = msgs.data[0].content
        text_out = []
        for p in content_parts:
            if getattr(p, "type", None) == "text":
                text_out.append(p.text.value)
        if text_out:
            return "\n".join(text_out)

    return "(No response returned from Assistants API)"

# ================= Section 3: Run & Review =================
st.markdown("## 3) Run Tests")
st.markdown("Click **Run Prompts** to test the prompts with your document as context.")

run = st.button("▶️ Run Prompts", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = []

if run:
    if not system_instructions.strip():
        st.warning("Please provide system instructions.")
    elif prompts_df.empty or not prompts_df["prompt"].dropna().tolist():
        st.warning("Please provide at least one prompt.")
    elif not doc_text.strip():
        st.warning("Please upload and parse a document to use as context.")
    else:
        results = []
        chunks = chunk_text(doc_text, max_chars=chunk_chars)
        progress = st.progress(0, text="Starting...")
        total = len([p for p in prompts_df["prompt"].fillna("").tolist() if p.strip()])
        completed = 0

        for _, row in prompts_df.iterrows():
            prompt = (row.get("prompt") or "").strip()
            expected = (row.get("expected_result") or "").strip()
            if not prompt:
                continue

            # --------- NEW: Top-K selection per prompt ---------
            query_for_ranking = f"{prompt} {system_instructions or ''}"
            selected = select_top_k_chunks(chunks, query_for_ranking, k=top_k)
            context = ("\n\n---\n\n").join(selected)
            # Safety clamp to avoid extremely long contexts
            if len(context) > 12000:
                context = context[:12000]

            try:
                if api_key:
                    if backend == "Assistants (beta)":
                        ai_text = call_openai_assistants(system_instructions, context, prompt)
                    else:
                        user_message = f"Context from uploaded document:\n\n{context}\n\nUser prompt:\n{prompt}"
                        messages = [
                            {"role": "system", "content": system_instructions or ""},
                            {"role": "user", "content": user_message},
                        ]
                        ai_text = call_openai_completions(messages)
                else:
                    ai_text = "(Demo response) This is a placeholder. With an API key, the model would answer using the document context and your prompt."
            except TypeError as e:
                if "proxies" in str(e):
                    ai_text = ("ERROR: HTTP client compatibility issue detected. "
                               "Pin httpx==0.27.2 in requirements.txt and reinstall.")
                else:
                    ai_text = f"ERROR: {e}"
            except Exception as e:
                ai_text = f"ERROR: {e}"

            results.append({
                "document": doc_name or "",
                "system_instructions": system_instructions,
                "prompt": prompt,
                "expected_result": expected,
                "ai_response": ai_text
            })

            completed += 1
            progress.progress(min(int(completed / max(total, 1) * 100), 100),
                              text=f"Processed {completed} of {total}")

        st.session_state["results"] = results
        st.success("Completed. Review results below.")

# ======== Report builders ================
def build_markdown_report(title, system_instructions, doc_name, results):
    lines = [
        f"# {title}",
        "",
        f"**Document:** {doc_name or '—'}",
        "",
        "## System Instructions",
        "",
        system_instructions or "_None_",
        "",
        "## Results",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"### {i}. {r.get('prompt','')}",
            "",
            "**AI Response**",
            "",
            r.get("ai_response",""),
            "",
        ]
        if r.get("expected_result"):
            lines += ["**Expected Result (notes)**", "", r["expected_result"], ""]
    return "\n".join(lines)

def build_html_report(title, system_instructions, doc_name, results):
    css = """
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;line-height:1.55;color:#222;
           max-width:980px;margin:2rem auto;padding:0 1rem;}
      h1,h2,h3{margin:.4rem 0 0.6rem}
      .meta{color:#555;margin:.3rem 0 1rem;font-size:.95rem}
      .card{border:1px solid #eaecef;border-radius:12px;padding:16px;margin:16px 0;
            box-shadow:0 1px 2px rgba(0,0,0,.04)}
      pre{white-space:pre-wrap;background:#f6f8fa;border:1px solid #eaecef;border-radius:8px;
          padding:12px;margin:.5rem 0}
      code{font-family:ui-monospace,SFMono-Regular,Consolas,Monaco,monospace}
      .small{font-size:.9rem;color:#444}
    </style>
    """
    def esc(x): return html.escape(x or "")
    parts = [f"<h1>{esc(title)}</h1>",
             f"<div class='meta'><b>Document:</b> {esc(doc_name or '—')}</div>",
             "<h2>System Instructions</h2>",
             f"<pre>"+esc(system_instructions or 'None')+"</pre>",
             "<h2>Results</h2>"]
    for i, r in enumerate(results, 1):
        parts.append("<div class='card'>")
        parts.append(f"<h3>{i}. {esc(r.get('prompt',''))}</h3>")
        parts.append(f"<div class='small'><b>Expected Result (notes):</b> {esc(r.get('expected_result',''))}</div>")
        parts.append("<h4>AI Response</h4>")
        parts.append("<pre>"+esc(r.get('ai_response',''))+"</pre>")
        parts.append("</div>")
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title>{css}</head><body>{''.join(parts)}</body></html>"
    return html_doc.encode("utf-8")

def build_docx_report(title, system_instructions, doc_name, results):
    d = Document()
    d.add_heading(title, 0)
    d.add_paragraph(f"Document: {doc_name or '—'}")
    d.add_heading("System Instructions", level=1)
    d.add_paragraph(system_instructions or "None")
    d.add_heading("Results", level=1)
    for i, r in enumerate(results, 1):
        d.add_heading(f"{i}. {r.get('prompt','')}", level=2)
        d.add_paragraph("AI Response:")
        d.add_paragraph(r.get("ai_response",""))
        if r.get("expected_result"):
            d.add_paragraph("Expected Result (notes):")
            d.add_paragraph(r["expected_result"])
    bio = BytesIO()
    d.save(bio)
    return bio.getvalue()

# ================= Section 4: Results & Export =================
st.markdown("## 4) Results & Export")

if st.session_state.get("results"):
    results = st.session_state["results"]
    df = pd.DataFrame(results)

    tabs = st.tabs(["🗂 Cards", "📊 Table", "🖨 Report Preview"])

    with tabs[0]:
        st.markdown("#### Results (readable cards)")
        for r in results:
            title = (r.get("prompt") or "")[:120] or "(no prompt)"
            with st.expander(f"🧪 {title}"):
                st.markdown(f"**Document:** {doc_name or '—'}")
                if r.get("expected_result"):
                    st.markdown("**Expected Result (notes):**")
                    st.markdown(r["expected_result"])
                st.markdown("**AI Response:**")
                st.markdown(r.get("ai_response", ""))

    with tabs[1]:
        st.markdown("#### Raw table")
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("#### Print-friendly Markdown (preview)")
        _md_preview = build_markdown_report(
            "BidWERX Prompt Lab — Test Run",
            system_instructions,
            doc_name,
            results
        )
        st.markdown(_md_preview)

    # ---------- Downloads ----------
    st.markdown("### Download")

    md_text   = build_markdown_report("BidWERX Prompt Lab — Test Run", system_instructions, doc_name, results)
    html_bytes = build_html_report("BidWERX Prompt Lab — Test Run", system_instructions, doc_name, results)
    docx_bytes = build_docx_report("BidWERX Prompt Lab — Test Run", system_instructions, doc_name, results)

    # CSV/JSON
    if "CSV" in export_format:
        csv_bytes = df.to_csv(index=False, quoting=csv.QUOTE_ALL).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes,
                           file_name="prompt_lab_results.csv", mime="text/csv")
    if "JSON" in export_format:
        json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button("⬇️ Download JSON", data=json_bytes,
                           file_name="prompt_lab_results.json", mime="application/json")

    # Extra report formats
    st.download_button("⬇️ Download HTML Report", data=html_bytes,
                       file_name="prompt_lab_report.html", mime="text/html")
    st.download_button("⬇️ Download Markdown", data=md_text.encode("utf-8"),
                       file_name="prompt_lab_report.md", mime="text/markdown")
    st.download_button("⬇️ Download DOCX (Word)", data=docx_bytes,
                       file_name="prompt_lab_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # ZIP handoff kit (everything in one)
    cfg = {
        "system_instructions": system_instructions,
        "prompts": prompts_df.to_dict(orient="records"),
        "document": doc_name or "",
    }
    kit_bytes_io = io.BytesIO()
    with zipfile.ZipFile(kit_bytes_io, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prompts.json", json.dumps(cfg, indent=2))
        zf.writestr("results.csv", df.to_csv(index=False, quoting=csv.QUOTE_ALL))
        zf.writestr("results.json", df.to_json(orient="records", indent=2))
        zf.writestr("report.md", md_text)
        zf.writestr("report.html", html_bytes)
    st.download_button("📦 Download Handoff Kit (ZIP)",
                       data=kit_bytes_io.getvalue(),
                       file_name="bidwerx_prompt_handoff.zip",
                       mime="application/zip")
else:
    st.info("Run prompts to see results here.")

with st.expander("ℹ️ Notes & Tips"):
    st.markdown("""
    - Retrieval is enabled: the app selects **Top-K** relevant chunks per prompt.
    - Toggle **Backend** between **Completions** and **Assistants (beta)** in the sidebar.
    - For scanned PDFs, consider adding OCR if parsing returns little/no text.
    - API key is session-only here; for production, use environment variables or a secrets manager.
    """)
