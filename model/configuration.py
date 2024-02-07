import warnings
import torch
import numpy as np
from accelerate import Accelerator
from nltk.translate.bleu_score import sentence_bleu
from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')


def prepare_lora_configuration():
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
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


def prepare_training_arguments(config):
    training_args = TrainingArguments(
        per_device_train_batch_size=config.args_training.train_batch_size,
        per_device_eval_batch_size=config.args_training.eval_batch_size,
        auto_find_batch_size=True,
        num_train_epochs=config.args_training.num_train_epochs,
        learning_rate=config.args_training.learning_rate,
        save_total_limit=config.args_training.save_total_limit,
        load_best_model_at_end=True,
        logging_steps=config.args_training.logging_steps,
        output_dir=config.args_training.dir_checkpoint,
        save_strategy=config.args_training.save_strategy,
        evaluation_strategy=config.args_training.evaluation_strategy,
        optim=config.args_training.optimizer,
        deepspeed=config.deepspeed.stage_2,
        bf16=True,
        report_to=config.args_training.report_to,
        run_name=config.args_training.run_name
    )
    return training_args


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, model_name):
    preds, labels = eval_preds
    tokenizer = prepare_tokenizer(model_name)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    references = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
    generated_texts = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    bleu_scores_ngram_1 = []
    bleu_scores_ngram_2 = []
    bleu_scores_ngram_3 = []
    bleu_scores_ngram_4 = []
    bleu_scores_ngram_avg = []
    for reference_text, generated_text in zip(references, generated_texts):
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
    tokenizer.padding_side = "left"
    return tokenizer


def prepare_model(model_name):
    tokenizer = prepare_tokenizer(model_name)
    bnb_config = prepare_quantization_configuration()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return tokenizer, model
