import warnings
import torch
import hydra
from load_model import get_model, get_tokenizer
from gradio_chat import launch_gradio_chat
warnings.filterwarnings('ignore')


@hydra.main(config_path='../config', config_name='model_checkpoint', version_base=None)
def main(config):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, streamer = get_tokenizer(config.model.best_checkpoint)
    model = get_model(config.model.best_checkpoint)
    launch_gradio_chat(tokenizer, streamer, model, device)


if __name__ == '__main__':
    main()
