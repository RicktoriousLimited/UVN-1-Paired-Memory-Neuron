# UVN-1 Paired-Memory Neuron

Exact mathematical implementation of the **UVN-1 Paired-Memory Neuron** architecture, combined with a fully differentiable PyTorch version that can be trained as a conversation chatbot.

## Architecture Summary

- **4 neurons**, each with an **8-dimensional private state**
- **16-dimensional** byte/character embedding
- Pairwise message passing with learned attention (`edge_logits` + `U_ij`)
- **3-branch non-linear mixture**
- Gated state update (GRU-style)
- **Paired outer-product memory** matrix `F_t` per neuron
- Population readout to next-character logits

## Files

| File | Description |
|------|-------------|
| `pytorch_uvn1_chatbot.py` | Fully differentiable PyTorch implementation + training + chat |
| `original_equations.md` | The exact mathematical equations from the original specification |
| `architecture_diagram.md` | Description of the architecture |

## Quick Start

```bash
pip install torch
python pytorch_uvn1_chatbot.py
```

## Results (after training)

```
You : hello
UVN : hi how are you a: i am d

You : thank you
UVN : you are welcome.

You : goodbye
UVN : see you later.
```

## Origin

This repository combines:
1. The original exact mathematical UVN-1 specification
2. A clean, modern PyTorch implementation that makes the entire architecture trainable end-to-end
3. A simple conversation chatbot trained on the architecture

Built collaboratively in an interactive research session.
