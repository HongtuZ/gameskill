import os

# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '2048'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '2048'
os.environ['FPS_MAX_FRAMES'] = '20'

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from pathlib import Path

adapter_dir = 'output/Qwen3.5-4B-lmdb/checkpoint-1070'
enable_thinking = True

model, processor = get_model_processor('Qwen/Qwen3.5-4B')  # attn_impl='flash_attention_2'
model = PeftModel.from_pretrained(model, adapter_dir)
template = get_template(processor, enable_thinking=enable_thinking)
engine = TransformersEngine(model, template=template)

GAME_PROMPT = Path("prompts/valorant.md").read_text(encoding="utf-8")
infer_request = InferRequest(
    messages=[
        {
            "role": "system",
            "content": f"你是一个专业的FPS游戏实时教练。请根据视频片段分析玩家操作并结合具体游戏机制给出建议。以下游戏知识供参考：\n\n{GAME_PROMPT}\n\n",
        },
        {"role": "user", "content": "<video>\n请分析这段游戏视频片段，判断是否需要立即给出实时指导。"},
    ],
    videos=["027.mp4"],
)
request_config = RequestConfig(max_tokens=1024, temperature=0.2)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
# request_config = RequestConfig(max_tokens=1024, temperature=0, stream=True)
# gen_list = engine.infer([infer_request], request_config=request_config)
# for chunk in gen_list[0]:
#     if chunk is None:
#         continue
#     print(chunk.choices[0].delta.content, end='', flush=True)
# print()

