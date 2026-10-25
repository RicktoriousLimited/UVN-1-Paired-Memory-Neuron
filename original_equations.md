# UVN-1 Exact Mathematical Equations

These are the original equations that define the UVN-1 Paired-Memory Neuron.

## 1. Input Byte Embedding
```
x_t = E_bytes(b_t)  ∈ R^E     (E = 16)
```

## 2. Pairwise Message Passing
```
a_ij = softmax_j(edge_i)
m_i  = Σ_j a_ij · U_ij · s_j(t-1)
```

## 3. Context Concatenation
```
q_i = [x_t ; s_i(t-1) ; m_i]  ∈ R^32
```

## 4. Branch Mixture
```
v_ik = B_ikᵀ · q_i + b_ik
p_ik = softmax_k(mix_i)
u_i  = tanh( Σ_k p_ik · tanh(v_ik) )
```

## 5. Gated State Update
```
r_i     = sigmoid(G_iᵀ · q_i + g_i)
s_i(t)  = r_i ⊙ s_i(t-1) + (1 - r_i) ⊙ u_i
```

## 6. Paired Memory Update
```
F_t = (1 - λ) · F_{t-1} + η_t · (v_t · k_tᵀ)
```
where `k_t = tanh(s_i(t))`, `v_t = u_i`, `η_t = mean(r_i)`

## 7. Population Readout
```
z_t = W_readout · [s_1 ; s_2 ; s_3 ; s_4] + bias
```

## Dimensions
- N = 4 neurons
- D = 8 (private state width)
- E = 16 (embedding)
- K = 3 (mixture branches)
- Q_DIM = 32
