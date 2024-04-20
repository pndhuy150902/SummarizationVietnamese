import warnings
import torch
import numpy as np
from accelerate import Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AwqConfig

warnings.filterwarnings('ignore')


def prepare_lora_configuration():
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
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
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config


def prepare_quantization_configuration():
    # awq_config = AwqConfig(
    #     bits=4,
    #     group_size=128,
    #     zero_point=True
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def prepare_training_arguments(config):
    training_args = TrainingArguments(
        per_device_train_batch_size=config.args_training.train_batch_size,
        per_device_eval_batch_size=config.args_training.eval_batch_size,
        # auto_find_batch_size=config.args_training.auto_find_batch_size,
        num_train_epochs=config.args_training.num_train_epochs,
        learning_rate=config.args_training.learning_rate,
        weight_decay=config.args_training.weight_decay,
        lr_scheduler_type=config.args_training.lr_scheduler_type,
        save_total_limit=config.args_training.save_total_limit,
        # load_best_model_at_end=config.args_training.load_best_model_at_end,
        gradient_accumulation_steps=config.args_training.gradient_accumulation_steps,
        gradient_checkpointing=config.args_training.gradient_checkpointing,
        logging_steps=config.args_training.logging_steps,
        output_dir=config.args_training.dir_checkpoint,
        save_strategy=config.args_training.save_strategy,
        evaluation_strategy=config.args_training.evaluation_strategy,
        optim=config.args_training.optimizer,
        deepspeed=config.deepspeed.stage_2,
        bf16=config.args_training.bf16,
        report_to=config.args_training.report_to,
        run_name=config.args_training.run_name
    )
    return training_args


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, rouge_metric, model_name):
    preds, labels = eval_preds
    tokenizer = prepare_tokenizer(model_name)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, rouge_types=['rouge1', 'rouge2', 'rougeL'])
    return {k: round(v, 4) for k, v in rouge_scores.items()}


def prepare_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def prepare_model(model_name):
    tokenizer = prepare_tokenizer(model_name, token="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")
    bnb_config = prepare_quantization_configuration()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        token="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"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return tokenizer, model
