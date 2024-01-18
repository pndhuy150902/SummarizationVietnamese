import warnings
import torch
import hydra
from nltk.translate.bleu_score import sentence_bleu
from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')


def prepare_lora_configuration():
    lora_config = LoraConfig(
        r=32,
        alpha=64,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
            'lm_head',
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config


def prepare_quantization_configuration():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def prepare_training_arguments():
    training_args = TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=30,
        learning_rate=1e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir='./model_checkpoint',
        save_strategy='epoch',
        optim='paged_adamw_8bit',
        bf16=True,
    )
    return training_args


def compute_metrics(pred, train_dataset):
    references = pred.label_ids
    generated_texts = pred.predictions
    bleu_scores_ngram_1 = []
    bleu_scores_ngram_2 = []
    bleu_scores_ngram_3 = []
    bleu_scores_ngram_4 = []
    bleu_scores_ngram_avg = []
    for reference, generated_text in zip(references, generated_texts):
        reference_text = train_dataset[reference]['text']
        bleu_score_ngram_1 = sentence_bleu([reference_text], generated_text, weights=(1, 0, 0, 0))
        bleu_score_ngram_2 = sentence_bleu([reference_text], generated_text, weights=(0, 1, 0, 0))
        bleu_score_ngram_3 = sentence_bleu([reference_text], generated_text, weights=(0, 0, 1, 0))
        bleu_score_ngram_4 = sentence_bleu([reference_text], generated_text, weights=(0, 0, 0, 1))
        bleu_score_ngram_avg = sentence_bleu([reference_text], generated_text, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores_ngram_1.append(bleu_score_ngram_1)
        bleu_scores_ngram_2.append(bleu_score_ngram_2)
        bleu_scores_ngram_3.append(bleu_score_ngram_3)
        bleu_scores_ngram_4.append(bleu_score_ngram_4)
        bleu_scores_ngram_avg.append(bleu_score_ngram_avg)

    return {
        'bleu@1': sum(bleu_scores_ngram_1) / len(bleu_scores_ngram_1),
        'bleu@2': sum(bleu_scores_ngram_2) / len(bleu_scores_ngram_2),
        'bleu@3': sum(bleu_scores_ngram_3) / len(bleu_scores_ngram_3),
        'bleu@4': sum(bleu_scores_ngram_4) / len(bleu_scores_ngram_4),
        'bleu@avg': sum(bleu_scores_ngram_avg) / len(bleu_scores_ngram_avg)
    }


def prepare_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


@hydra.main(config_path='../config/model', config_name='pretrained_model', version_base=None)
def prepare_model(config):
    tokenizer = prepare_tokenizer(config.model_name)
    bnb_config = prepare_quantization_configuration()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model