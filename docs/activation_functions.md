# Activation Functions — A Complete Guide

---

## Why Do We Need Activation Functions?

A neural network without activation functions is just a stack of linear
transformations. No matter how many linear layers you chain, the result
collapses into a single linear operation:

```
layer2(layer1(x)) = W2 × (W1 × x + b1) + b2
                  = (W2 × W1) × x + (W2 × b1 + b2)
                  = W_eff × x + b_eff          ← still linear!
```

Activation functions inject **non-linearity** between layers, allowing the
network to learn complex, curved decision boundaries — essentially anything
beyond a straight line.

---

## 1. Sigmoid (Logistic)

### Formula

```
σ(x) = 1 / (1 + e^(-x))
```

### Graph

```
  1.0 ┤                              ●●●●●●●●●●
      │                         ●●●●
      │                      ●●●
      │                    ●●
  0.5 ┤                  ●
      │                ●●
      │             ●●●
      │          ●●●●
  0.0 ┤●●●●●●●●●
      └──────────────────────────────────────────
     -6          -3          0          3          6
```

### Properties

| Property       | Value                   |
| -------------- | ----------------------- |
| Output range   | (0, 1)                  |
| Centered at    | 0.5 (not zero-centered) |
| Derivative max | 0.25 at x = 0           |
| Monotonic      | Yes                     |
| Differentiable | Yes, everywhere         |
| Derivative     | σ(x) × (1 − σ(x))       |

### Neural Network Performance

**Strengths:**

- Outputs can be interpreted as probabilities (0 to 1).
- Smooth gradient — no discontinuities.

**Weaknesses:**

- **Vanishing gradients**: The maximum derivative is only 0.25. In deep
  networks, gradients shrink exponentially through layers
  (0.25 × 0.25 × 0.25 ...), making early layers nearly impossible to train.
- **Not zero-centered**: Outputs are always positive, which causes
  zig-zagging gradient updates (all weights in a layer shift the same
  direction).
- **Expensive**: `exp()` is computationally heavier than simpler functions.

**When to use:** Output layer for binary classification (probability of
class 1). Rarely used in hidden layers of modern networks.

---

## 2. Tanh (Hyperbolic Tangent)

### Formula

```
tanh(x) = (e^x − e^(-x)) / (e^x + e^(-x))
```

Equivalently: `tanh(x) = 2σ(2x) − 1`

### Graph

```
 +1.0 ┤                              ●●●●●●●●●●
      │                         ●●●●
      │                      ●●●
      │                    ●●
  0.0 ┤──────────────────●──────────────────────
      │                ●●
      │             ●●●
      │          ●●●●
 -1.0 ┤●●●●●●●●●
      └──────────────────────────────────────────
     -4          -2          0          2          4
```

### Properties

| Property       | Value             |
| -------------- | ----------------- |
| Output range   | (−1, +1)          |
| Centered at    | 0 (zero-centered) |
| Derivative max | 1.0 at x = 0      |
| Monotonic      | Yes               |
| Differentiable | Yes, everywhere   |
| Derivative     | 1 − tanh²(x)      |

### Neural Network Performance

**Strengths:**

- **Zero-centered**: Outputs span negative and positive, so gradient
  updates aren't biased in one direction.
- Stronger gradients than sigmoid (max derivative is 1.0 vs 0.25).

**Weaknesses:**

- **Still saturates**: For large |x|, the derivative approaches 0.
  Vanishing gradients remain a problem in very deep networks.
- `exp()` computation cost.

**When to use:** Hidden layers in recurrent networks (LSTMs, GRUs).
Output layer when you need values in [-1, 1]. Used in PPO's actor-critic
networks because it constrains activations to a stable range.

---

## 3. ReLU (Rectified Linear Unit)

### Formula

```
ReLU(x) = max(0, x)
```

### Graph

```
      │                          ●
      │                        ●
      │                      ●
      │                    ●
      │                  ●
  0.0 ┤●●●●●●●●●●●●●●●●
      │
      └──────────────────────────────────────────
     -4          -2          0          2          4
```

### Properties

| Property       | Value                                     |
| -------------- | ----------------------------------------- |
| Output range   | [0, +∞)                                   |
| Centered at    | Not zero-centered                         |
| Derivative     | 0 for x < 0, 1 for x > 0 (undefined at 0) |
| Monotonic      | Yes                                       |
| Differentiable | No (kink at x = 0, but works in practice) |

### Neural Network Performance

**Strengths:**

- **No vanishing gradient** for positive values — derivative is exactly 1.
- **Extremely fast**: Just a comparison and a copy. No `exp()`.
- **Sparse activation**: Negative inputs produce 0, meaning many neurons
  are "off" at any time. This sparsity is computationally efficient and
  can act as implicit regularization.
- Enables training of very deep networks (100+ layers).

**Weaknesses:**

- **Dying ReLU**: If a neuron's input is always negative (e.g., after a
  bad weight update), its output is permanently 0 and its gradient is
  permanently 0. The neuron is "dead" — it can never recover.
- **Not zero-centered**: Like sigmoid, outputs are always ≥ 0.
- **Unbounded**: Outputs can grow arbitrarily large, which may cause
  instability without proper normalization.

**When to use:** The default choice for hidden layers in feedforward
networks and CNNs. If in doubt, start with ReLU.

---

## 4. Leaky ReLU

### Formula

```
LeakyReLU(x) = x        if x ≥ 0
               α × x    if x < 0      (α = 0.01 typically)
```

### Graph

```
      │                          ●
      │                        ●
      │                      ●
      │                    ●
      │                  ●
  0.0 ┤            ≈≈≈●              ← slight negative slope
      │         ≈≈≈
      └──────────────────────────────────────────
     -4          -2          0          2          4
```

### Properties

| Property       | Value                         |
| -------------- | ----------------------------- |
| Output range   | (−∞, +∞)                      |
| Slope (x > 0)  | 1                             |
| Slope (x < 0)  | α (small constant, e.g. 0.01) |
| Monotonic      | Yes                           |
| Differentiable | No (kink at 0)                |

### Neural Network Performance

**Strengths:**

- **No dying neurons**: Even for negative inputs, there's a small gradient
  (α), so weights can still update and recover.
- Fast computation, like ReLU.

**Weaknesses:**

- The slope α is a hyperparameter. Results can be sensitive to its value.
- Not zero-centered.

**When to use:** Drop-in replacement for ReLU when you observe dying
neurons (large portions of activations stuck at 0).

---

## 5. Parametric ReLU (PReLU)

### Formula

```
PReLU(x) = x        if x ≥ 0
           α × x    if x < 0      (α is LEARNED, not fixed)
```

### Properties

Same shape as Leaky ReLU, but α is a trainable parameter. The network
learns the optimal negative slope during backpropagation.

### Neural Network Performance

**Strengths:**

- Adapts the negative slope to the data — more flexible than Leaky ReLU.
- Shown to improve accuracy on ImageNet by ~1% over ReLU.

**Weaknesses:**

- Adds learnable parameters (one per channel in CNNs).
- Risk of overfitting on small datasets.

**When to use:** Large-scale image classification where slight accuracy
gains matter.

---

## 6. ELU (Exponential Linear Unit)

### Formula

```
ELU(x) = x                   if x ≥ 0
         α × (e^x − 1)       if x < 0      (α = 1.0 typically)
```

### Graph

```
      │                          ●
      │                        ●
      │                      ●
      │                    ●
      │                  ●
  0.0 ┤              ●●●
      │          ●●●●
 -α   ┤- - ●●●●- - - - - - - - - - - - - - - -
      │
      └──────────────────────────────────────────
     -4          -2          0          2          4
```

### Properties

| Property      | Value                                      |
| ------------- | ------------------------------------------ |
| Output range  | (−α, +∞)                                   |
| Zero-centered | Approximately (mean activation close to 0) |
| Smooth        | Yes (no kink at 0)                         |
| Monotonic     | Yes                                        |

### Neural Network Performance

**Strengths:**

- **Negative saturation**: For large negative inputs, output approaches
  −α, making the mean activation closer to 0 (reduces bias shift).
- **Smooth at 0**: Unlike ReLU, no kink. This can help optimization.
- Pushes mean activations toward zero (like batch normalization, but free).

**Weaknesses:**

- `exp()` is slower than ReLU's simple comparison.
- Still saturates for very negative x (gradient → 0), though this is
  less problematic than sigmoid because it only affects one side.

**When to use:** When you want benefits of ReLU but need activations
closer to zero-mean without batch normalization.

---

## 7. SELU (Scaled Exponential Linear Unit)

### Formula

```
SELU(x) = λ × x                   if x ≥ 0
           λ × α × (e^x − 1)       if x < 0

where λ ≈ 1.0507, α ≈ 1.6733  (derived mathematically)
```

### Properties

| Property         | Value                                       |
| ---------------- | ------------------------------------------- |
| Output range     | (−λα, +∞) ≈ (−1.758, +∞)                    |
| Self-normalizing | Yes — activations converge to mean 0, var 1 |

### Neural Network Performance

**Strengths:**

- **Self-normalizing**: With proper weight initialization (LeCun normal)
  and specific architecture constraints, activations automatically
  converge to zero mean and unit variance across layers. No batch
  normalization needed.
- Enables training very deep feedforward networks without normalization layers.

**Weaknesses:**

- Only works with specific conditions: fully-connected layers, LeCun
  normal initialization, no skip connections.
- Breaks if you use dropout (must use AlphaDropout instead).
- Fragile — deviating from the required setup loses the self-normalizing property.

**When to use:** Deep feedforward networks (no convolutions, no skip
connections) where you want to avoid batch normalization.

---

## 8. GELU (Gaussian Error Linear Unit)

### Formula

```
GELU(x) = x × Φ(x)

where Φ(x) is the CDF of the standard normal distribution.

Approximation: GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

### Graph

```
      │                          ●●
      │                        ●●
      │                      ●●
      │                    ●●
      │                 ●●●
  0.0 ┤           ●●●●●
      │         ●●
 -0.17┤- - - ●●- - - - - - - - - - - - - - - -
      │     ●●●●●●●●
  0.0 ┤●●●●●
      └──────────────────────────────────────────
     -4          -2          0          2          4
```

### Properties

| Property      | Value                                     |
| ------------- | ----------------------------------------- |
| Output range  | ≈ (−0.17, +∞)                             |
| Smooth        | Yes, everywhere                           |
| Non-monotonic | Yes (slight dip below 0 around x ≈ −0.75) |

### Neural Network Performance

**Strengths:**

- **Smooth approximation of ReLU** that allows small negative values
  through, acting as a soft gate.
- Combines properties of ReLU (linear for large positive x) with
  dropout-like stochastic regularization (small inputs have a chance
  of being zeroed).
- **State-of-the-art in transformers**: Used in BERT, GPT, and most
  modern language models.

**Weaknesses:**

- More expensive than ReLU (involves Φ or its tanh approximation).
- Marginal benefit over ReLU in simple/small networks.

**When to use:** Transformer architectures (NLP, vision transformers).
The default activation in modern large-scale models.

---

## 9. Swish / SiLU (Sigmoid Linear Unit)

### Formula

```
Swish(x) = x × σ(x) = x / (1 + e^(-x))
```

### Graph

```
      │                          ●●
      │                        ●●
      │                      ●●
      │                    ●●
      │                 ●●●
  0.0 ┤           ●●●●●
      │         ●●
 -0.28┤- - - ●- - - - - - - - - - - - - - - - -
      │    ●●●●●●●●
  0.0 ┤●●●●
      └──────────────────────────────────────────
     -6          -3          0          3          6
```

### Properties

| Property      | Value                               |
| ------------- | ----------------------------------- |
| Output range  | ≈ (−0.278, +∞)                      |
| Smooth        | Yes, everywhere                     |
| Non-monotonic | Yes (dips below 0 around x ≈ −1.28) |
| Derivative    | σ(x) + x × σ(x) × (1 − σ(x))        |

### Neural Network Performance

**Strengths:**

- Discovered via neural architecture search (Google, 2017).
- Consistently outperforms ReLU on deep networks (ImageNet, CIFAR).
- Smooth, non-monotonic — provides a self-gating mechanism.
- Unbounded above, bounded below.

**Weaknesses:**

- Slower than ReLU (requires sigmoid computation).
- Slight extra memory for storing intermediate sigmoid values.

**When to use:** EfficientNet and modern CNN architectures. Good general
replacement for ReLU when compute budget allows it.

---

## 10. Mish

### Formula

```
Mish(x) = x × tanh(softplus(x)) = x × tanh(ln(1 + e^x))
```

### Graph

Similar shape to Swish but slightly different curvature in the negative region.

```
      │                          ●●
      │                        ●●
      │                      ●●
      │                    ●●
      │                 ●●●
  0.0 ┤           ●●●●●
      │         ●●
 -0.31┤- - - ●- - - - - - - - - - - - - - - - -
      │    ●●●●●●●●
  0.0 ┤●●●●
      └──────────────────────────────────────────
     -6          -3          0          3          6
```

### Properties

| Property      | Value                          |
| ------------- | ------------------------------ |
| Output range  | ≈ (−0.309, +∞)                 |
| Smooth        | Yes, infinitely differentiable |
| Non-monotonic | Yes                            |

### Neural Network Performance

**Strengths:**

- Slightly outperforms Swish on some vision tasks (YOLOv4).
- Very smooth loss landscape — helps optimization converge.
- Self-regularizing due to the bounded negative region.

**Weaknesses:**

- Computationally expensive (tanh + softplus).
- Gains over Swish are marginal and task-dependent.

**When to use:** Object detection (used in YOLOv4). Worth trying when
Swish already outperforms ReLU in your task.

---

## 11. Softmax

### Formula

```
softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
```

### Example

```
logits:        [2.0,  1.0,  0.1]
exp(logits):   [7.39, 2.72, 1.11]
sum:           11.22
softmax:       [0.66, 0.24, 0.10]    ← sums to 1.0
```

### Properties

| Property       | Value                           |
| -------------- | ------------------------------- |
| Output range   | (0, 1) per element              |
| Sum of outputs | Exactly 1.0                     |
| Use case       | Output layer only (multi-class) |

### Neural Network Performance

**Strengths:**

- Produces a valid probability distribution.
- Differentiable, works cleanly with cross-entropy loss.

**Weaknesses:**

- Not used in hidden layers (would force a sum-to-1 constraint
  on intermediate representations, destroying information).
- Exponentials can overflow for large logits (mitigated by subtracting max).

**When to use:** **Only** at the output layer for multi-class
classification. In PPO, softmax is implicitly applied via
`Categorical(logits=...)`.

---

## Comparison Table

| Function   | Range       | Zero-centered | Dying Neurons | Compute Cost | Best For                         |
| ---------- | ----------- | ------------- | ------------- | ------------ | -------------------------------- |
| Sigmoid    | (0, 1)      | No            | No            | Medium       | Binary output layer              |
| Tanh       | (−1, +1)    | Yes           | No            | Medium       | RNNs, PPO hidden layers          |
| ReLU       | [0, +∞)     | No            | Yes           | Very low     | Default for CNNs/MLPs            |
| Leaky ReLU | (−∞, +∞)    | No            | No            | Very low     | When ReLU neurons are dying      |
| PReLU      | (−∞, +∞)    | No            | No            | Low          | Large-scale image classification |
| ELU        | (−α, +∞)    | ~Yes          | No            | Medium       | When you need ~zero-mean outputs |
| SELU       | (−λα, +∞)   | Yes (auto)    | No            | Medium       | Deep FC networks, no batch norm  |
| GELU       | (−0.17, +∞) | ~Yes          | No            | Medium       | Transformers (BERT, GPT)         |
| Swish/SiLU | (−0.28, +∞) | ~Yes          | No            | Medium       | EfficientNet, modern CNNs        |
| Mish       | (−0.31, +∞) | ~Yes          | No            | High         | Object detection (YOLOv4)        |
| Softmax    | (0, 1) each | No            | N/A           | Medium       | Multi-class output layer only    |

---

## Decision Flowchart

```
Start
  │
  ├─ Output layer?
  │    ├─ Binary classification → Sigmoid
  │    ├─ Multi-class classification → Softmax
  │    ├─ Regression (unbounded) → Linear (no activation)
  │    └─ Regression (bounded [-1,1]) → Tanh
  │
  └─ Hidden layer?
       ├─ Transformer model → GELU
       ├─ RNN / LSTM → Tanh
       ├─ Deep FC, no batch norm → SELU
       ├─ Modern CNN (compute ok) → Swish or Mish
       └─ General purpose
            ├─ Start with ReLU
            ├─ Neurons dying? → Leaky ReLU or ELU
            └─ Need zero-mean activations? → ELU
```

---

## Vanishing Gradient — The Core Problem

The **vanishing gradient problem** is the single biggest reason activation
function choice matters. During backpropagation, gradients are multiplied
across layers:

```
∂Loss/∂W₁ = ∂Loss/∂out × ∂out/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁ × ∂h₁/∂W₁
                          ├────────────────────────────────┤
                          Each of these includes the activation's derivative
```

If each derivative is < 1 (sigmoid: max 0.25, tanh: max 1.0), the product
shrinks exponentially:

```
Sigmoid:  0.25⁵ = 0.001   → 5 layers deep, gradient is 1/1000th
Tanh:     0.65⁵ = 0.116   → better, but still shrinks
ReLU:     1.0⁵  = 1.0     → perfect (for active neurons)
```

This is why ReLU and its variants dominate modern deep learning — they
maintain gradient flow through many layers.

---

## PyTorch Usage

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.Tanh(),          # or nn.ReLU(), nn.GELU(), nn.SiLU(), etc.
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 2),   # output layer — no activation (logits)
)
```

All activation functions in PyTorch:

```python
nn.Sigmoid()
nn.Tanh()
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.PReLU(num_parameters=1)
nn.ELU(alpha=1.0)
nn.SELU()
nn.GELU()
nn.SiLU()           # Swish
nn.Mish()
nn.Softmax(dim=-1)  # output layer only
```
