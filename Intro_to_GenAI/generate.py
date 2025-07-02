# generate.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 small model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generation parameters
max_new_tokens = 50
top_k = 50
temperatures = [0.7, 1.0]

# Store results
results = []

for temp in temperatures:
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append((temp, generated_text))

# Save to file
with open("generated_samples.txt", "w", encoding="utf-8") as f:
    for temp, text in results:
        f.write(f"--- Temperature: {temp} ---\n")
        f.write(text + "\n\n")
