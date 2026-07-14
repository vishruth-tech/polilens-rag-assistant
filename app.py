import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "## 📚 Relevant Context\n\n"

    for doc in context:
        source = doc.metadata.get("source", "Unknown source")

        result += f"**Source:** `{source}`\n\n"
        result += doc.page_content + "\n\n"
        result += "---\n\n"

    return result


def chat(history):
    # Latest user question
    last_message = history[-1]["content"]

    # Previous conversation history
    prior = history[:-1]

    # Call RAG pipeline
    answer, context = answer_question(
        last_message,
        prior
    )

    # Add assistant's answer
    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return history, format_context(context)


def main():

    def put_message_in_chatbot(message, history):
        return "", history + [
            {
                "role": "user",
                "content": message
            }
        ]

    with gr.Blocks(
        title="PoliLens Expert Assistant"
    ) as ui:

        gr.Markdown(
            "# 🏢 PoliLens Expert Assistant\n"
            "Ask me anything about PoliLens!"
        )

        with gr.Row():

            with gr.Column():

                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600
                )

                message = gr.Textbox(
                    placeholder="Ask anything about PoliLens..."
                )

            with gr.Column():

                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*"
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot]
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown]
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()