import warnings
import gradio as gr
from threading import Thread
warnings.filterwarnings('ignore')


def clear_all(msg_box, content_box):
    return None, None


def clear_content(content_box):
    return None


def launch_gradio_chat(tokenizer, streamer, model, device):
    with gr.Blocks() as app:
        with gr.Row():
            msg_box = gr.Chatbot(height=768)
            def user(user_message, history):
                return "", history + [[user_message, None]]
            def bot(history):
                news_prompt = f"""<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt:
{str(history[-1][0])} [/INST] """
                inputs = tokenizer(news_prompt, add_special_tokens=True, return_tensors="pt").to(device)
                dict_kwargs = dict(inputs, early_stopping=False, max_new_tokens=1024, temperature=0.2, top_p=0.3, top_k=50, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id, streamer=streamer)
                t = Thread(target=model.generate, kwargs=dict_kwargs)
                t.start()
                history[-1][1] = ""
                for character in streamer:
                    history[-1][1] += character
                    yield history
        with gr.Row():
            with gr.Column(scale=5):
                content_box = gr.Textbox(label='Nhập nội dung mà bạn muốn tóm tắt')
                content_box.submit(user, inputs=[content_box, msg_box], outputs=[content_box, msg_box], queue=False).then(bot, msg_box, msg_box)
            with gr.Column(scale=1):
                btn_submit = gr.Button(value='SUBMIT', size='sm')
                btn_submit.click(user, inputs=[content_box, msg_box], outputs=[content_box, msg_box], queue=False).then(bot, msg_box, msg_box)
                btn_clear = gr.ClearButton(value='CLEAR', size='sm')
                btn_clear_all = gr.ClearButton(value='CLEAR ALL', size='sm')
                btn_clear.click(clear_content, inputs=[content_box], outputs=[content_box])
                btn_clear_all.click(clear_all, inputs=[msg_box, content_box], outputs=[msg_box, content_box])
    app.queue()
    app.launch(share=True, debug=True)
