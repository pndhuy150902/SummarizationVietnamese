import warnings
from vllm import LLM, SamplingParams
warnings.filterwarnings('ignore')

TEMPLATE = """<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm tắt ngắn gọn nội dung sau bằng tiếng Việt:
{} [/INST]
"""


def create_prompt(context):
    full_prompt = TEMPLATE.format(context)
    return full_prompt


def setup_vllm():
    llm = LLM(model='./')
    sample_params = SamplingParams(max_tokens=12288,
                                   temperature=0.7,
                                   skip_special_tokens=True)
    return llm, sample_params
