"""
UVN-1 Paired-Memory Neuron
- Trains on unstructured text
- Supports online / continual learning while chatting
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class UVN1(nn.Module):
    def __init__(self, vocab_size, n_neurons=4, d_state=8, e_dim=16, k_mixtures=3):
        super().__init__()
        self.N, self.D, self.E, self.K = n_neurons, d_state, e_dim, k_mixtures
        self.Q_DIM = e_dim + d_state + d_state
        self.LAMBDA = 0.05

        self.embed = nn.Embedding(vocab_size, e_dim)
        self.edge_logits = nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        self.U = nn.Parameter(torch.randn(self.N, self.N, self.D, self.D) * 0.1)
        self.B = nn.Parameter(torch.randn(self.N, self.K, self.Q_DIM, self.D) * 0.1)
        self.b = nn.Parameter(torch.randn(self.N, self.K, self.D) * 0.1)
        self.mix_logits = nn.Parameter(torch.randn(self.N, self.K) * 0.1)
        self.G = nn.Parameter(torch.randn(self.N, self.Q_DIM, self.D) * 0.1)
        self.g = nn.Parameter(torch.randn(self.N, self.D) * 0.1)
        self.readout = nn.Linear(self.N * self.D, vocab_size)

    def step(self, char_idx, s):
        x = self.embed(char_idx)
        a = torch.softmax(self.edge_logits, dim=-1)
        m = torch.zeros(self.D, self.N)
        for i in range(self.N):
            for j in range(self.N):
                m[:, i] = m[:, i] + a[i, j] * (self.U[i, j] @ s[:, j])

        s_new = torch.zeros(self.D, self.N)
        for i in range(self.N):
            q = torch.cat([x, s[:, i], m[:, i]])
            p = torch.softmax(self.mix_logits[i], dim=-1)
            u = torch.zeros(self.D)
            for k in range(self.K):
                v = q @ self.B[i, k] + self.b[i, k]
                u = u + p[k] * torch.tanh(v)
            u = torch.tanh(u)
            r = torch.sigmoid(q @ self.G[i] + self.g[i])
            s_new[:, i] = r * s[:, i] + (1 - r) * u
        logits = self.readout(s_new.reshape(-1))
        return s_new, logits

    def forward(self, seq):
        s = torch.zeros(self.D, self.N)
        outputs = []
        for t in range(seq.size(0)):
            s, logits = self.step(seq[t], s)
            outputs.append(logits)
        return torch.stack(outputs)


# -------------------- Unstructured training data --------------------
unstructured_text = """
hello how are you? i am doing well thank you.
my name is uvn one. i am a small paired memory neuron.
i can chat with you and learn from any text you give me.
thank you. you are welcome.
goodbye. see you later. have a nice day.
what can you do? i can learn from unstructured data and improve while we talk.
this is free form text. the neuron reads it character by character and updates itself.
learning from unstructured data makes the model more flexible.
"""

# Also keep some dialogue style data
dialogue_data = [
    "hello a: hi how are you?",
    "how are you a: i am doing well thank you.",
    "what is your name a: my name is uvn one.",
    "who are you a: i am a small paired memory neuron.",
    "thank you a: you are welcome.",
    "goodbye a: goodbye have a nice day.",
    "what can you do a: i can chat and learn from text.",
]

all_text = unstructured_text + " ".join(dialogue_data)
chars = sorted(list(set(all_text + " ")))
vocab_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_vocab = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)

print(f"Vocabulary size: {VOCAB_SIZE}")
print("Training on unstructured + dialogue data...\n")

model = UVN1(VOCAB_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.015)
criterion = nn.CrossEntropyLoss()

def train_on_text(text, epochs=30):
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        # chunk the text
        seq = torch.tensor([vocab_to_idx[c] for c in text if c in vocab_to_idx], dtype=torch.long)
        if len(seq) < 2:
            return
        optimizer.zero_grad()
        logits = model(seq[:-1])
        loss = criterion(logits, seq[1:])
        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{epochs} | Loss: {total_loss:.4f}")

# 1. Train on unstructured text
print("[1] Learning from unstructured text...")
train_on_text(unstructured_text, epochs=40)

# 2. Fine-tune on dialogue style
print("\n[2] Fine-tuning on dialogue patterns...")
for text in dialogue_data:
    train_on_text(text, epochs=5)

print("\nTraining finished.\n")

# -------------------- Online learning chat --------------------
def chat_with_online_learning(user_input, online_lr=0.005):
    model.eval()
    prompt = user_input.lower().strip() + " a:"
    s = torch.zeros(model.D, model.N)

    # Prime with prompt
    with torch.no_grad():
        for ch in prompt:
            idx = vocab_to_idx.get(ch, 0)
            s, _ = model.step(torch.tensor(idx), s)

    reply = ""
    generated_indices = []

    with torch.no_grad():
        for _ in range(30):
            logits = model.readout(s.reshape(-1))
            next_idx = torch.argmax(logits).item()
            ch = idx_to_vocab.get(next_idx, " ")
            reply += ch
            generated_indices.append(next_idx)
            s, _ = model.step(torch.tensor(next_idx), s)
            if ch in ".?!" and len(reply) > 4:
                break

    # ----- Online learning: update from this conversation -----
    model.train()
    full = prompt + reply
    seq = torch.tensor([vocab_to_idx.get(c, 0) for c in full], dtype=torch.long)
    if len(seq) > 3:
        optimizer.zero_grad()
        # temporary higher lr for online step
        for g in optimizer.param_groups:
            g['lr'] = online_lr
        logits = model(seq[:-1])
        loss = criterion(logits, seq[1:])
        loss.backward()
        optimizer.step()
        for g in optimizer.param_groups:
            g['lr'] = 0.015   # restore

    return reply.strip()

print("=== Chat with Online Learning (UVN-1) ===\n")
print("(The model improves a little after every reply)\n")

test_queries = [
    "hello",
    "how are you",
    "what is your name",
    "thank you",
    "what can you do",
    "goodbye",
]

for q in test_queries:
    response = chat_with_online_learning(q)
    print(f"You : {q}")
    print(f"UVN : {response}\n")
