import warnings
from gradio_chat import gradio_chat
warnings.filterwarnings('ignore')


def main():
    app = gradio_chat()
    app.launch(share=True)


if __name__ == '__main__':
    main()
