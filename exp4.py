import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import textwrap

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=200, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Explicitly create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Run text generation
prompt = "Once upon a time"
generated_text = generate_text(prompt)

# Print neatly (no horizontal scroll)
print("\n".join(textwrap.wrap(generated_text, width=100)))
