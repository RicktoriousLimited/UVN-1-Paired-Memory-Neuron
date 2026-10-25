"""
UVN-1 Paired-Memory Neuron - Fully Differentiable PyTorch Implementation
+ Conversation Chatbot

This combines the exact UVN-1 architecture with modern PyTorch training.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class PyTorchUVN1Bot(nn.Module):
    def __init__(self, vocab_size, n_neurons=4, d_state=8, e_dim=16, k_mixtures=3):
        super().__init__()
        self.N = n_neurons
        self.D = d_state
        self.E = e_dim
        self.K = k_mixtures
        self.Q_DIM = e_dim + d_state + d_state
        self.LAMBDA = 0.05

        # Embedding
        self.E_bytes = nn.Embedding(vocab_size, e_dim)

        # Pairwise routing + transformation
        self.edge_logits = nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        self.U = nn.Parameter(torch.randn(self.N, self.N, self.D, self.D) * 0.1)

        # Mixture branches
        self.B = nn.Parameter(torch.randn(self.N, self.K, self.Q_DIM, self.D) * 0.1)
        self.b = nn.Parameter(torch.randn(self.N, self.K, self.D) * 0.1)
        self.mix_logits = nn.Parameter(torch.randn(self.N, self.K) * 0.1)

        # Gate
        self.G = nn.Parameter(torch.randn(self.N, self.Q_DIM, self.D) * 0.1)
        self.g = nn.Parameter(torch.randn(self.N, self.D) * 0.1)

        # Readout
        self.W_readout = nn.Linear(self.N * self.D, vocab_size)

    def forward(self, char_sequence):
        device = char_sequence.device
        s = torch.zeros(self.D, self.N, device=device)
        outputs = []

        for t in range(char_sequence.size(0)):
            x = self.E_bytes(char_sequence[t])

            # Pairwise messages
            a = torch.softmax(self.edge_logits, dim=-1)
            m = torch.zeros(self.D, self.N, device=device)
            for i in range(self.N):
                for j in range(self.N):
                    m[:, i] = m[:, i] + a[i, j] * (self.U[i, j] @ s[:, j])

            s_new = torch.zeros(self.D, self.N, device=device)
            for i in range(self.N):
                q = torch.cat([x, s[:, i], m[:, i]])

                # 3-branch mixture
                p = torch.softmax(self.mix_logits[i], dim=-1)
                u = torch.zeros(self.D, device=device)
                for k in range(self.K):
                    v = q @ self.B[i, k] + self.b[i, k]
                    u = u + p[k] * torch.tanh(v)
                u = torch.tanh(u)

                # Gated update
                r = torch.sigmoid(q @ self.G[i] + self.g[i])
                s_new[:, i] = r * s[:, i] + (1 - r) * u

            logits = self.W_readout(s_new.reshape(-1))
            outputs.append(logits)
            s = s_new

        return torch.stack(outputs)


# -------------------- Training Data --------------------
data = [
    "hello a: hi how are you?",
    "hi a: hello nice to meet you.",
    "how are you a: i am doing well thank you.",
    "what is your name a: my name is uvn one.",
    "who are you a: i am a small paired memory neuron.",
    "thank you a: you are welcome.",
    "goodbye a: goodbye have a nice day.",
    "bye a: see you later.",
    "what can you do a: i can chat with you.",
    "help a: just type a message and i will reply.",
    "how old are you a: i was just trained.",
    "nice to meet you a: nice to meet you too.",
]

chars = sorted(list(set("".join(data) + " ")))
vocab_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_vocab = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)

print(f"Vocabulary size: {VOCAB_SIZE}")
print("Training UVN-1 Paired-Memory Neuron...\n")

model = PyTorchUVN1Bot(VOCAB_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(50):
    total = 0.0
    for text in data:
        optimizer.zero_grad()
        seq = torch.tensor([vocab_to_idx[c] for c in text], dtype=torch.long)
        logits = model(seq[:-1])
        loss = criterion(logits, seq[1:])
        loss.backward()
        optimizer.step()
        total += loss.item()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/50 | Loss: {total/len(data):.4f}")

print("\nTraining finished.\n")

# -------------------- Chat --------------------
def chat(user):
    model.eval()
    prompt = user.lower().strip() + " a:"
    with torch.no_grad():
        s = torch.zeros(model.D, model.N)
        for ch in prompt:
            idx = vocab_to_idx.get(ch, 0)
            x = model.E_bytes(torch.tensor(idx))
            a = torch.softmax(model.edge_logits, dim=-1)
            m = torch.zeros(model.D, model.N)
            for i in range(model.N):
                for j in range(model.N):
                    m[:, i] = m[:, i] + a[i, j] * (model.U[i, j] @ s[:, j])
            s_new = torch.zeros(model.D, model.N)
            for i in range(model.N):
                q = torch.cat([x, s[:, i], m[:, i]])
                p = torch.softmax(model.mix_logits[i], dim=-1)
                u = torch.zeros(model.D)
                for k in range(model.K):
                    v = q @ model.B[i, k] + model.b[i, k]
                    u = u + p[k] * torch.tanh(v)
                u = torch.tanh(u)
                r = torch.sigmoid(q @ model.G[i] + model.g[i])
                s_new[:, i] = r * s[:, i] + (1 - r) * u
            s = s_new

        reply = ""
        for _ in range(25):
            logits = model.W_readout(s.reshape(-1))
            next_idx = torch.argmax(logits).item()
            ch = idx_to_vocab.get(next_idx, " ")
            reply += ch

            x = model.E_bytes(torch.tensor(next_idx))
            a = torch.softmax(model.edge_logits, dim=-1)
            m = torch.zeros(model.D, model.N)
            for i in range(model.N):
                for j in range(model.N):
                    m[:, i] = m[:, i] + a[i, j] * (model.U[i, j] @ s[:, j])
            s_new = torch.zeros(model.D, model.N)
            for i in range(model.N):
                q = torch.cat([x, s[:, i], m[:, i]])
                p = torch.softmax(model.mix_logits[i], dim=-1)
                u = torch.zeros(model.D)
                for k in range(model.K):
                    v = q @ model.B[i, k] + model.b[i, k]
                    u = u + p[k] * torch.tanh(v)
                u = torch.tanh(u)
                r = torch.sigmoid(q @ model.G[i] + model.g[i])
                s_new[:, i] = r * s[:, i] + (1 - r) * u
            s = s_new

            if ch in ".?!" and len(reply) > 3:
                break
    return reply.strip()

print("=== UVN-1 Conversation Demo ===\n")
for q in ["hello", "how are you", "what is your name", "thank you", "goodbye", "who are you"]:
    print(f"You : {q}")
    print(f"UVN : {chat(q)}\n")
