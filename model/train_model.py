import warnings
import hydra
from trl import SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForLanguageModeling
from configuration import prepare_lora_configuration, prepare_training_arguments, prepare_model, compute_metrics, preprocess_logits_for_metrics
from prepare_dataset import prepare_dataset

warnings.filterwarnings('ignore')


def prepare_trainer(config):
    lora_config = prepare_lora_configuration()
    training_args = prepare_training_arguments(config)
    tokenizer, model = prepare_model(config.model_name_test)
    func_collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1.0)
    dataset = prepare_dataset(config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        dataset_text_field='text',
        data_collator=func_collate,
        compute_metrics=lambda x: compute_metrics(x, config.model_name_test),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stop_callback],
        packing=False
    )
    return trainer


@hydra.main(config_path='../config', config_name='hyperparameters', version_base=None)
def main(config):
    trainer = prepare_trainer(config)
    trainer.train()
    
    
if __name__ == '__main__':
    main()
