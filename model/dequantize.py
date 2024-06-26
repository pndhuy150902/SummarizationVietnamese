import os
import json
import copy
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft.utils import _get_submodules
from peft import PeftModel

def dequantize_model(model, to='./dequantized_model', dtype=torch.bfloat16, device="cuda"):
    os.makedirs(to, exist_ok=True)

    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        # a hack, setting this to avoid hf's saving error because hf
        # itself does not support saving a model that is registered to be loaded in 4bit.
        model.is_loaded_in_4bit = False

        print("Saving dequantized model...")
        model.save_pretrained(to)
        #tokenizer.save_pretrained(to)
        config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
        config_data.pop("quantization_config", None)
        config_data.pop("pretraining_tp", None)
        with open(os.path.join(to, 'config.json'), 'w') as config:
            config.write(json.dumps(config_data, indent=2))

        return model

if __name__ == "__main__":
    adapter = "./model_checkpoint/checkpoint-1002"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Viet-Mistral/Vistral-7B-Chat", token="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")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Viet-Mistral/Vistral-7B-Chat", quantization_config=bnb_config, attn_implementation="flash_attention_2", token="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", device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = dequantize_model(model, to='./dqz_model_qdora/',dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained("./model_vistral_merged_qdora_v2/")
    tokenizer.save_pretrained("./model_vistral_merged_qdora_v2/")
