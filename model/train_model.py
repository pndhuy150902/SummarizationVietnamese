import warnings
import hydra
import torch
import evaluate
import wandb
from peft import get_peft_model, set_peft_model_state_dict, load_peft_weights
from trl import SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForLanguageModeling
from configuration import prepare_lora_configuration, prepare_training_arguments, prepare_model, compute_metrics, preprocess_logits_for_metrics
from prepare_dataset import prepare_dataset

warnings.filterwarnings('ignore')


def prepare_trainer(config):
    lora_config = prepare_lora_configuration()
    training_args = prepare_training_arguments(config)
    tokenizer, model = prepare_model(config.model_mistral)
    func_collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # early_stop_callback = EarlyStoppingCallback(early_stopping_patience=2)
    dataset = prepare_dataset(config)
    rouge_metric = evaluate.load("rouge")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field='text',
        max_seq_length=config.length.text,
        data_collator=func_collate,
        dataset_num_proc=config.args_training.dataset_num_proc,
        compute_metrics=lambda x: compute_metrics(x, rouge_metric, config.model_mistral),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # callbacks=[early_stop_callback],
        packing=False
    )
    return trainer, tokenizer


@hydra.main(config_path='../config', config_name='hyperparameters', version_base=None)
def main(config):
    torch.manual_seed(42)
    trainer, tokenizer = prepare_trainer(config)
    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()
