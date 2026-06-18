import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs) -> str:
    """Format retrieved context chunks for the side panel."""
    if not docs:
        return "No context retrieved."
    result = ""
    for i, doc in enumerate(docs, 1):
        source   = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "")
        result  += (
            f"📄 [{i}] {doc_type.upper()} — {source}\n\n"
            f"{doc.page_content}\n\n"
            f"{'─' * 60}\n\n"
        )
    return result


def chat(message: str, history: list):
    """Called on every user submission."""
    if not message.strip():
        return "", history, ""

    history = history or []

    answer, docs = answer_question(message, history)

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]

    return "", history, format_context(docs)


def clear_all():
    return [], ""


def main():
    with gr.Blocks(title="TalentBridge SA Assistant") as ui:  # theme moved out

        gr.Markdown(
            "# 🏢 TalentBridge SA — Talent Assistant\n"
            "Ask me to **find graduates**, **shortlist candidates**, "
            "or learn about **our services**."
        )

        with gr.Row():
            # ── Left: chat ─────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=520,
                    # 'type' and 'bubble_full_width' removed — not supported
                    # in older Gradio versions
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="e.g. Find graduates suitable for a finance internship …",
                        show_label=False,
                        scale=5,
                    )
                    submit_btn = gr.Button("Submit", scale=1, variant="secondary")

            # ── Right: retrieved context ────────────────────────────────────
            with gr.Column(scale=2):
                context_box = gr.Textbox(
                    label="📚 Retrieved Context",
                    lines=30,
                    interactive=False,
                )

        gr.Examples(
            examples=[
                "Find graduates suitable for a finance internship.",
                "Who has experience with Power BI and Accounting?",
                "Recommend marketing graduates with digital marketing certifications.",
                "Find civil engineers available immediately.",
                "Which graduates have project management skills?",
                "Create a shortlist for an HR Administrator role.",
                "What TalentBridge service helps companies build project teams?",
                "Compare available candidates for a Business Analyst position.",
            ],
            inputs=message,
            label="💡 Try these sample queries",
        )

        # ── Event wiring ────────────────────────────────────────────────────
        message.submit(
            fn=chat,
            inputs=[message, chatbot],
            outputs=[message, chatbot, context_box],
        )
        submit_btn.click(
            fn=chat,
            inputs=[message, chatbot],
            outputs=[message, chatbot, context_box],
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()