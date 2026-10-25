# UVN-1 Paired-Memory Neuron

Exact mathematical UVN-1 architecture + fully differentiable PyTorch implementation.

Now supports:
- **Unstructured data training** (any free-form text)
- **Online / continual learning** (improves while chatting)

## Files

| File | Description |
|------|-------------|
| `uvn1_unstructured_online.py` | **Main file** – trains on unstructured text + online learning chat |
| `pytorch_uvn1_chatbot.py` | Earlier structured-dialogue version |
| `original_equations.md` | Exact mathematical equations of UVN-1 |

## Quick Start

```bash
pip install torch
python uvn1_unstructured_online.py
```

## What is new

1. **Unstructured data**  
   The model first reads a free-form paragraph of text and learns next-character prediction from it.

2. **Online learning**  
   After every reply during chat, the model performs a small gradient update on the conversation it just had. It slowly improves from the interaction itself.

## Example Output

```
You : hello
UVN : hi how are you?

You : thank you
UVN : you are welcome.

You : what can you do
UVN : i can chat and learn from text.
```

## Architecture (reminder)

- 4 neurons × 8-D private state
- Pairwise message passing
- 3-branch non-linear mixture
- Gated state update
- Paired outer-product memory F_t
- Population readout

Built collaboratively in an interactive research session.
