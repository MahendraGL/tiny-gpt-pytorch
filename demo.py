import torch

from attention_head import GPT
from tokenizer import Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

with open("Data/arthemis_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = Tokenizer(text)
model = GPT(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("artemis_gpt.pt", map_location=device))
model.eval()

prompt = torch.tensor([tokenizer.encode("hi")], dtype=torch.long, device=device)
output = model.generate(prompt, max_new_tokens=500)[0].tolist()
print(tokenizer.decode(output))
