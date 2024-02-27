import warnings
from vllm import LLM, SamplingParams
warnings.filterwarnings('ignore')

TEMPLATE = """<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm tắt ngắn gọn nội dung sau bằng tiếng Việt:
{} [/INST]"""


def create_prompt(context):
    full_prompt = TEMPLATE.format(context)
    return full_prompt


def setup_vllm():
    llm = LLM(model='./')
    generation_kwargs = dict('text', streamer=True, temperature=0.7, max_tokens=1024, top_p=0.9, repetition_penalty=1.2)
    sample_params = SamplingParams(max_tokens=1024,
                                   temperature=0.7,
                                   skip_special_tokens=True)
    return llm, sample_params
