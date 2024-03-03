import warnings
import gradio as gr
warnings.filterwarnings('ignore')


def clear_all(msg_box, content_box):
    return None, None


def clear_content(content_box):
    return None


def respond_chat(message, history):
    bot_message = "Xin lỗi vì mô hình chưa thể sẵn sàng để sử dụng. Mong bạn quay lại sau."
    history.append((message, bot_message))
    return '', history


def launch_gradio_chat(tokenizer, streamer, model):
    with gr.Blocks() as app:
        with gr.Row():
            msg_box = gr.Chatbot(height=768)
        with gr.Row():
            with gr.Column(scale=5):
                content_box = gr.Textbox(label='Nhập nội dung mà bạn muốn tóm tắt')
                content_box.submit(respond_chat, inputs=[content_box, msg_box], outputs=[content_box, msg_box])
            with gr.Column(scale=1):
                btn_submit = gr.Button(value='SUBMIT', size='sm')
                btn_submit.click(respond_chat, inputs=[content_box, msg_box], outputs=[content_box, msg_box])
                btn_clear = gr.ClearButton(value='CLEAR', size='sm')
                btn_clear_all = gr.ClearButton(value='CLEAR ALL', size='sm')
                btn_clear.click(clear_content, inputs=[content_box], outputs=[content_box])
                btn_clear_all.click(clear_all, inputs=[msg_box, content_box], outputs=[msg_box, content_box])
    app.queue()
    app.launch(share=True, debug=True)
