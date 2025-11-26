from transformers import AutoTokenizer
from models.modeling_deepseek_v3_mobe import DeepseekV3MoBEForCausalLM 
from models.modeling_qwen3_mobe import Qwen3MoBEForCausalLM
from models.modeling_kimi_k2_mobe import KimiK2MoBEForCausalLM
import torch

# model_name = "/root/DeepSeek-V3-0324-MoBE"
model_name = "../Qwen/MoBE/Qwen3-30B-A3B-Instruct-2507-MoBE"
offload_folder = "./offload_dir"


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

max_memory = {i: "40GiB" for i in range(2)}  
max_memory["cpu"] = "400GiB"      

if 'Qwen' in model_name:
    model = Qwen3MoBEForCausalLM.from_pretrained(
    		model_name,
    		device_map="auto",
    		offload_folder=offload_folder,
    		offload_state_dict=True,
    		torch_dtype=torch.bfloat16,
        max_memory=max_memory
    )
elif 'DeepSeek' in model_name:
    model = DeepseekV3MoBEForCausalLM.from_pretrained(
    		model_name,
    		device_map="auto",
    		offload_folder=offload_folder,
    		offload_state_dict=True,
    		torch_dtype=torch.bfloat16,
        max_memory=max_memory
    )
else:
		model = KimiK2MoBEForCausalLM.from_pretrained(
    		model_name,
    		device_map="auto",
    		offload_folder=offload_folder,
    		offload_state_dict=True,
    		torch_dtype=torch.bfloat16,
        max_memory=max_memory
    )

input_text = "Artificial intelligence is"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)