import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load the LLaMA 7B model and tokenizer from Hugging Face with 8-bit quantization
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quantization_config, 
#     device_map="auto"
# )

model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)


# Function to generate text based on a prompt
def generate_text(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation_config = {
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
    outputs = model.generate(**inputs, **generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)