import matplotlib.pyplot as plt
import torch

from attention_head import GPT
from config import batch_size, block_size, eval_interval, eval_iters, learning_rate, max_iters
from tokenizer import Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

with open("Data/arthemis_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = Tokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
validation_data = data[split:]


def get_batch(data_split):
    source = train_data if data_split == "train" else validation_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[start:start + block_size] for start in starts])
    y = torch.stack([source[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    results = {}
    model.eval()

    for data_split in ["train", "validation"]:
        total_loss = 0
        for _ in range(eval_iters):
            x, y = get_batch(data_split)
            _, loss = model(x, y)
            total_loss += loss.item()
        results[data_split] = total_loss / eval_iters

    model.train()
    return results


steps = []
training_losses = []
validation_losses = []

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        steps.append(step)
        training_losses.append(losses["train"])
        validation_losses.append(losses["validation"])
        print(
            f"step {step}: train loss {losses['train']:.4f}, "
            f"validation loss {losses['validation']:.4f}"
        )

    x, y = get_batch("train")
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "artemis_gpt.pt")
print("\nSaved model checkpoint: artemis_gpt.pt")

plt.plot(steps, training_losses, label="Training loss")
plt.plot(steps, validation_losses, label="Validation loss")
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("Artemis GPT training")
plt.legend()
plt.tight_layout()
plt.savefig("training_metrics.png")
plt.close()
print("Saved training plot: training_metrics.png")

model.eval()
prompt = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
output = model.generate(prompt, max_new_tokens=500)[0].tolist()
print("\nGenerated text:\n")
print(tokenizer.decode(output))
