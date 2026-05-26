# app.py — Gradio web UI for the RAG PDF Chatbot
# Handles: file upload → PDF indexing → question input → answer display

import gradio as gr       # Gradio — builds interactive web UIs in pure Python
import os                 # for path operations

# Import our custom RAG pipeline functions
from rag_engine import load_and_index_pdf, build_qa_chain

# ── Global State ─────────────────────────────────────────────────────────────
# Gradio calls each function fresh per interaction, so we store the chain here
# between the "upload PDF" step and the "ask question" step
qa_chain = None      # will hold the RetrievalQA chain once PDF is indexed
vector_store = None  # will hold the FAISS index after PDF is processed


def process_pdf(pdf_file) -> str:
    """
    Triggered when user clicks 'Index PDF'.
    Loads the uploaded PDF, builds the FAISS index, and sets up the QA chain.

    Args:
        pdf_file: Gradio file object — has .name attribute pointing to temp file path

    Returns:
        str: Status message shown in the UI text box
    """
    global qa_chain, vector_store  # write to module-level state

    # Guard: user clicked the button without uploading a file
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."

    try:
        # pdf_file.name is the path Gradio saved the upload to (temp directory)
        print(f"[INFO] Processing PDF: {pdf_file.name}")  # log for debugging

        # Step 1: Load PDF and build FAISS vector index
        vector_store = load_and_index_pdf(pdf_file.name)

        # Step 2: Build the LangChain QA chain over the indexed vector store
        qa_chain = build_qa_chain(vector_store)

        return "✅ PDF indexed successfully! You can now ask questions below."

    except FileNotFoundError:
        # Specific error if the temp file somehow disappeared
        return "❌ File not found. Please re-upload the PDF."

    except Exception as e:
        # Catch-all — show the error in UI rather than crashing silently
        return f"❌ Error processing PDF: {str(e)}"


def answer_question(question: str) -> str:
    """
    Triggered when user clicks 'Ask' or presses Enter.
    Runs the full RAG pipeline and returns a formatted answer.

    Args:
        question (str): The user's question from the text input

    Returns:
        str: Markdown-formatted answer with source page references
    """
    global qa_chain  # read the chain set during PDF upload

    # Guard: PDF hasn't been processed yet
    if qa_chain is None:
        return "⚠️ Please upload and index a PDF first."

    # Guard: empty or whitespace-only question
    if not question or not question.strip():
        return "⚠️ Please type a question before clicking Ask."

    try:
        # Run the full RAG pipeline:
        # question → FAISS retrieves top 3 chunks → LLM generates answer from chunks
        result = qa_chain({"query": question.strip()})

        answer = result["result"]  # the LLM's generated answer string

        # Extract page numbers from the source documents for transparency
        # source_documents is a list of Document objects with metadata
        sources = result.get("source_documents", [])
        source_pages = sorted(set(
            int(doc.metadata.get("page", 0)) + 1  # +1 because PDF pages are 0-indexed internally
            for doc in sources
            if "page" in doc.metadata  # only include docs that have page metadata
        ))

        # Build the formatted response
        if source_pages:
            # Show answer + which pages the context came from
            response = f"**Answer:**\n\n{answer}\n\n---\n📄 *Context retrieved from page(s): {source_pages}*"
        else:
            # Some PDFs don't have page metadata — still show the answer
            response = f"**Answer:**\n\n{answer}"

        return response

    except Exception as e:
        # Surface errors clearly so user knows what went wrong
        return f"❌ Error generating answer: {str(e)}"


# ── Gradio UI Layout ──────────────────────────────────────────────────────────
# Using gr.Blocks() for full layout control over rows, columns, and component placement

with gr.Blocks(
    title="RAG PDF Chatbot — LangChain + FAISS",  # browser tab title
    theme=gr.themes.Soft(),                        # clean, professional Gradio theme
    css=".gradio-container { max-width: 900px; margin: auto; }"  # center and cap width
) as demo:

    # ── Header Section ────────────────────────────────────────────────────────
    gr.Markdown("# 🤖 RAG PDF Chatbot")
    gr.Markdown(
        "> **Retrieval-Augmented Generation** — Upload any PDF, then ask questions. "
        "Answers are grounded in your document, not hallucinated.\n\n"
        "**Stack:** LangChain · FAISS Vector DB · HuggingFace Embeddings · Flan-T5 LLM · Gradio"
    )

    gr.Markdown("---")  # visual divider

    # ── Step 1: PDF Upload ────────────────────────────────────────────────────
    gr.Markdown("### Step 1 — Upload your PDF")

    with gr.Row():  # horizontal layout for file input + button + status
        with gr.Column(scale=2):
            pdf_input = gr.File(
                label="📄 PDF File",
                file_types=[".pdf"],    # restrict to PDF only for safety
                file_count="single"     # one file at a time
            )
        with gr.Column(scale=1):
            upload_btn = gr.Button(
                "📥 Index PDF",
                variant="primary",      # blue button — primary action
                size="lg"               # larger button for visibility
            )
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,      # read-only — just shows status messages
                placeholder="Status will appear here after upload..."
            )

    gr.Markdown("---")

    # ── Step 2: Question & Answer ─────────────────────────────────────────────
    gr.Markdown("### Step 2 — Ask questions about your PDF")

    question_input = gr.Textbox(
        label="💬 Your Question",
        placeholder="e.g. What is the main argument of this document?",
        lines=2,           # 2-line input box — comfortable for longer questions
        max_lines=5        # can expand up to 5 lines if needed
    )

    with gr.Row():
        ask_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
        clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")

    # Markdown component for the answer — supports bold/italic formatting
    answer_output = gr.Markdown(
        label="Answer",
        value="*Your answer will appear here...*"  # default placeholder text
    )

    # ── Example Questions ─────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["Summarize the key points."],
            ["What conclusions does the author draw?"],
            ["What methodology was used?"],
        ],
        inputs=question_input,   # clicking an example fills the question box
        label="💡 Example Questions"
    )

    # ── Wire Events ───────────────────────────────────────────────────────────

    # Upload button click → process_pdf function
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],        # pass the uploaded file object
        outputs=[upload_status]    # update the status textbox
    )

    # Ask button click → answer_question function
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],   # pass the typed question
        outputs=[answer_output]    # update the answer markdown area
    )

    # Enter key in question box also triggers the answer (better UX)
    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

    # Clear button resets the answer box back to placeholder
    clear_btn.click(
        fn=lambda: "*Your answer will appear here...*",  # lambda resets the markdown
        inputs=[],
        outputs=[answer_output]
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # share=False → local only (http://127.0.0.1:7860)
    # Change share=True when deploying to HuggingFace Spaces — generates a public URL
    demo.launch(
        share=False,       # set True for HuggingFace Spaces deployment
        show_error=True    # show full error tracebacks in the browser for debugging
    )
