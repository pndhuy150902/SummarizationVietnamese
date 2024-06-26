import warnings
import gc
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm.pandas()
gc.collect()
torch.manual_seed(42)


def create_prompt(sample):
    template = """<s>[INST] Hãy tóm tắt văn bản sau đây một cách ngắn gọn và chính xác. Đảm bảo rằng bạn tạo ra một bản tóm tắt trừu tượng và nêu được những ý chính, thông tin quan trọng nhất của văn bản. Dưới đây là văn bản cần tóm tắt:
{} [/INST]"""
    prompt = template.format(sample)
    return prompt


def generate_text():
  for batch_1, batch_2 in tqdm(zip(torch.utils.data.DataLoader(full_data_test['context'], batch_size=1, shuffle=False), torch.utils.data.DataLoader(full_data_test['summarization'], batch_size=1, shuffle=False), strict=True), total=int(round(len(full_data_test)/1, 0))):
    prompts = [create_prompt(context) for context in batch_1]
    inputs = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt").to(device)
    outputs = model.generate(
      **inputs,
      early_stopping=False,
      max_new_tokens=768,
      temperature=0.2,
      top_p=0.8,
      top_k=40,
      repetition_penalty=1.0,
      do_sample=True,
      pad_token_id=tokenizer.unk_token_id
    )
    references.extend(batch_2)
    predictions.extend([tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False).split('[/INST]')[1].strip() for output in outputs])
    torch.cuda.empty_cache()


if __name__ == "__main__":
    references = []
    predictions = []
    full_data_test = pd.read_csv('../dataset/val_tmp_loc_news.csv')
    qdora_merged = "./model_vistral_merged_qdora_v2/"
    model = AutoModelForCausalLM.from_pretrained(
        qdora_merged,
        device_map={"":0},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(qdora_merged)
    tokenizer.padding_side = "left"
    model.eval()
    generate_text()
    full_data_test['abstract_predictions'] = predictions
    rouge_metric = evaluate.load("rouge")
    rouge_scores = rouge_metric.compute(references=references, predictions=predictions, use_stemmer=True, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    print(rouge_scores)
    # full_data_test.to_csv('test_vistral_qdora_v2.csv', index=False)
