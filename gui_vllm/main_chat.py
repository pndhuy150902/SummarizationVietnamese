import warnings
import torch
from gradio_chat import gradio_chat
warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(42)
    app = gradio_chat()
    app.launch(share=True)


if __name__ == '__main__':
    main()
