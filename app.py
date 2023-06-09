from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, revision='main', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, revision='main', trust_remote_code=True).to('cuda')


text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

input_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')  # Moving input_ids to GPU

sequences = text_gen_pipeline(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")