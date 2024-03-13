import transformers,torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import AutoPeftModelForCausalLM

assert transformers.__version__ >= "4.34.1"

token="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"
# model_name="sanbongazin/willgpt-neox-small_v2"
# model_name="sanbongazin/willgpt-open-calm-1b"
model_name="sanbongazin/willgpt-Gemma2b"

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", low_cpu_mem_usage=True)
except:
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="cpu", low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,token=token)

prompt = "配信画面を可愛くするには"

token_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(
    input_ids=token_ids.to(model.device),
    max_new_tokens=150,
    # do_sample=True,
    temperature=0.8,
    streamer=streamer,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

output = tokenizer.decode(output_ids.tolist()[0])
print(prompt)
print(output)
