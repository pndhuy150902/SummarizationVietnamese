import warnings
from trl import SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForLanguageModeling
from configuration import prepare_lora_configuration, prepare_training_arguments, prepare_model

warnings.filterwarnings('ignore')


def prepare_trainer():
    lora_config = prepare_lora_configuration()
    training_args = prepare_training_arguments()
    tokenizer, model = prepare_model()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1.0)],
        packing=False
    )
    return trainer
