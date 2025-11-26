import transformers
import torch

model_id = "../Qwen/MoBE-32/Qwen3-30B-A3B-Instruct-2507-MoBE-hf"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant!"},
    {"role": "user", "content": "Can you explain the concept of regularization in machine learning?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
)
print(outputs[0]["generated_text"][-1]["content"])