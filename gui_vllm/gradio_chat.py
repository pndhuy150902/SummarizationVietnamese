import warnings
import gradio as gr
import numpy as np
warnings.filterwarnings('ignore')


def clear_all(msg_box, content_box):
    return None, None


def clear_content(content_box):
    return None


def respond_chat(message, history):
    bot_message = "Sorry that the model has not been prepared for text summarization."
    history.append((message, bot_message))
    return '', history


def gradio_chat():
    with gr.Blocks() as app:
        with gr.Row():
            msg_box = gr.Chatbot(height=768)
        with gr.Row():
            with gr.Column(scale=5):
                content_box = gr.Textbox(label='Enter your content which you want to summarize')
                content_box.submit(respond_chat, inputs=[content_box, msg_box], outputs=[content_box, msg_box])
            with gr.Column(scale=1):
                btn_submit = gr.Button(value='SUBMIT', size='sm')
                btn_submit.click(respond_chat, inputs=[content_box, msg_box], outputs=[content_box, msg_box])
                btn_clear = gr.ClearButton(value='CLEAR', size='sm')
                btn_clear_all = gr.ClearButton(value='CLEAR ALL', size='sm')
                btn_clear.click(clear_content, inputs=[content_box], outputs=[content_box])
                btn_clear_all.click(clear_all, inputs=[msg_box, content_box], outputs=[msg_box, content_box])
    return app
