import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

token="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"
model_name="sanbongazin/willgpt-neox-small_v2"


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


try:
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0",token=token)
except:
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0",token=token)

if torch.cuda.is_available():
    model = model.to("cuda")

text = "配信をかわいいく縁取りするには"
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=100,
        min_new_tokens=100,
        # do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)
