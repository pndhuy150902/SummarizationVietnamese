import warnings
import torch
from threading import Thread
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
warnings.filterwarnings('ignore')


def get_tokenizer_streamer(checkpoint):
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, token="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")
  tokenizer.padding_side = "left"
  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)
  return tokenizer, streamer


def get_model(checkpoint):
  config = PeftConfig.from_pretrained(checkpoint)
  base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map={"":0},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16
    ),
    token="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"
  )
  model = PeftModel.from_pretrained(base_model, checkpoint, device_map={"":0})
  model.eval()
  return model


def generate_text(prompt, tokenizer, streamer, model, history):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)
  dict_kwargs = dict(inputs, early_stopping=False, max_new_tokens=1024, temperature=0.2, top_p=0.3, top_k=50, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id, streamer=streamer)
  t = Thread(target=model.generate, kwargs=dict_kwargs)
  t.start()
  history[-1][1] = ''
  for new_text in streamer:
    history[-1][1] += new_text
    yield history
