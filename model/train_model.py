import warnings
import hydra
from trl import SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForLanguageModeling
from configuration import prepare_lora_configuration, prepare_training_arguments, prepare_model, compute_metrics, preprocess_logits_for_metrics

warnings.filterwarnings('ignore')


def prepare_trainer(config):
    lora_config = prepare_lora_configuration()
    training_args = prepare_training_arguments(config)
    tokenizer, model = prepare_model(config.model_name)
    func_collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1.0)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        data_collator=func_collate,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stop_callback],
        packing=False
    )
    return trainer


@hydra.main(config_path='../config/model', config_name='pretrained_model', version_base=None)
def main(config):
    trainer = prepare_trainer(config)
    trainer.train()
    
    
if __name__ == '__main__':
    main()
