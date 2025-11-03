# My Learning Journey: Implementing GPT-2 from Scratch

## Introduction

This document captures everything I learned while implementing a GPT-2 model completely from scratch. It's been a fascinating journey that allowed me to deeply understand how modern Large Language Models (LLMs) work.

## Fundamental Concepts Learned

### 1. Transformer Architecture

The first major lesson was understanding the Transformer architecture, introduced in the paper "Attention is All You Need". I learned that GPT-2 is based on:

- **Self-Attention Mechanism**: The heart of the model. Each token can "look at" other tokens and decide how relevant they are to its context.
- **Causal Masking**: In GPT-2, a token can only see the tokens that precede it, not future ones. This is essential for the language modeling task.
- **Multi-Head Attention**: Instead of a single attention mechanism, we use multiple ones (12 in the case of GPT-2 base) that operate in parallel, allowing the model to capture different types of relationships.

### 2. Self-Attention Implementation

I implemented Causal Self-Attention and understood that:

```
Q (Query), K (Key), V (Value) are computed from a single linear projection
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
```

**What I learned:**
- Division by sqrt(d_k) is fundamental to stabilize gradients
- Causal masking is achieved by masking with -inf the upper part of the attention matrix
- I used `torch.nn.functional.scaled_dot_product_attention` which optimizes this operation through kernel fusion (FlashAttention)

### 3. MLP (Multi-Layer Perceptron) Blocks

After each attention layer, there's an MLP that:
- Expands the embedding dimensions by a factor of 4x (768 → 3072 for GPT-2 base)
- Applies a GELU non-linearity (smoother than ReLU)
- Projects back to the original dimensions

**Important insight**: This step allows the model to process the information gathered from attention in a non-linear way.

### 4. Residual Connections and Layer Normalization

I understood the importance of skip connections (residual connections):
```
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

**Why they are crucial:**
- They allow gradients to flow directly through the network
- They prevent the vanishing gradient problem in deep networks
- LayerNorm before each block stabilizes training

### 5. Token and Position Embeddings

I learned that:
- **Token Embeddings (wte)**: Convert each token ID into a dense vector of dimension n_embd
- **Position Embeddings (wpe)**: Add information about the token's position in the sequence
- **Weight Tying**: Vaswani's paper suggests sharing weights between token embeddings and the final linear layer (lm_head), reducing total parameters

### 6. Weight Initialization

A critical aspect I learned is how to correctly initialize weights:

- Linear weights: `Normal(0, 0.02)`
- Residual weights: Scaled by `1/sqrt(2*n_layers)` to compensate for variance accumulation across layers
- Bias: Initialized to zero

**Why it's important**: Poor initialization can lead to exploding or vanishing gradients, making training impossible.

### 7. Optimization and Training

#### AdamW Optimizer
I implemented the AdamW optimizer with:
- Learning rate: 6e-4 (max)
- Betas: (0.9, 0.95)
- Weight decay: 0.1 (only for 2D weights, not for bias and LayerNorm)

**Important discovery**: Weight decay should not be applied to all parameters. Biases and 1D parameters (like those in LayerNorm) should not have weight decay.

#### Learning Rate Scheduling
I implemented a scheduler with:
1. **Linear warmup**: First 10 steps with linear LR increase
2. **Cosine decay**: After warmup, cosine decay down to min_lr = 0.1 * max_lr

**Why it works**: Warmup prevents initial gradients from being too large, while cosine decay allows the model to "refine itself" in the final training phases.

#### Gradient Accumulation
To simulate large batch sizes (524,288 tokens!) on limited hardware:
```
total_batch_size / (B * T * num_processes) = gradient_accumulation_steps
```

**What it does**: Accumulates gradients over multiple mini-batches before taking an optimizer step, simulating larger batches.

### 8. Mixed Precision Training

I used `torch.autocast` with `bfloat16`:
- **Advantages**: Reduces memory usage and accelerates training while maintaining numerical stability
- **bfloat16 vs float16**: bfloat16 has the same range as float32, so it's more stable

### 9. Distributed Data Parallel (DDP)

I learned how to parallelize training across multiple GPUs:
- Each process has a copy of the model
- Gradients are synchronized with `all_reduce` only when necessary
- The `require_backward_grad_sync` flag optimizes by avoiding unnecessary synchronizations during gradient accumulation

**Command to launch distributed training:**
```bash
torchrun --standalone --nproc_per_node=8 gpt2_trainer.py
```

### 10. Gradient Clipping

I implemented gradient clipping with maximum norm of 1.0:
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Why it's necessary**: Prevents gradient explosions that can completely destabilize training.

### 11. Data Loading and Tokenization

I created a custom DataLoader that:
- Uses tiktoken to tokenize text with GPT-2's vocabulary (50,257 tokens)
- Properly handles positioning for distributed training
- Implements wrapping when reaching the end of the dataset

**Important detail**: Each DDP process starts from a different position in the dataset to maximize batch diversity.

### 12. Inference and Generation

To generate text:
1. Encode the prompt
2. Autoregressive loop: predict the next token based on previous ones
3. Top-k sampling: Sample only from the k most likely tokens (k=50)
4. Multinomial sampling: Randomly choose among the top-k according to their probabilities

**Why top-k**: Prevents the model from choosing very unlikely tokens while maintaining variety in generation.

### 13. TF32 and Hardware Optimizations

I enabled TF32 on Ampere/Blackwell GPUs:
```python
torch.set_float32_matmul_precision('high')
```

**What it does**: Uses Tensor Float 32 for matrix multiplications, accelerating training without significant precision loss.

### 14. Model Compilation

I used `torch.compile(model)`:
- Optimizes the computational graph
- Fuses operations when possible
- Reduces Python overhead

**Result**: Noticeably faster training.

## Challenges Faced and Solutions

### Challenge 1: Out of Memory
**Problem**: The model wouldn't fit in GPU memory.
**Solution**: Gradient accumulation + batch size reduction + mixed precision training.

### Challenge 2: Unstable Training
**Problem**: Loss exploded after a few steps.
**Solution**: Proper weight initialization, gradient clipping, and learning rate warmup.

### Challenge 3: Slow Training
**Problem**: Training too slow to experiment effectively.
**Solution**: Enabled TF32, torch.compile, FlashAttention, and fused AdamW.

## GPT-2 Model Configurations

I learned the different sizes of GPT-2 models:

| Model | n_layers | n_heads | n_embd | Parameters |
|---------|----------|---------|--------|-----------|
| GPT-2   | 12       | 12      | 768    | 124M      |
| GPT-2-medium | 24   | 16      | 1024   | 350M      |
| GPT-2-large | 36    | 20      | 1280   | 774M      |
| GPT-2-xl | 48       | 25      | 1600   | 1558M     |

**Note**: All use vocab_size=50257 and block_size=1024.

## Metrics and Monitoring

During training I monitored:
- **Loss**: Cross-entropy loss (should decrease steadily)
- **Learning Rate**: Should follow the cosine schedule
- **Gradient Norm**: Should remain reasonable (< 1.0 after clipping)
- **Throughput**: Tokens processed per second (important for efficiency)
- **Time per Step**: Should be consistent

## Loading Pre-trained Weights

I implemented `from_pretrained` which:
1. Loads a HuggingFace GPT2LMHeadModel
2. Maps the HF model weights to my custom model
3. Handles the necessary transposition for OpenAI's Conv1D weights

**Technical detail**: OpenAI uses Conv1D instead of Linear, so weights need to be transposed.

## Vocabulary Padding

I noticed the vocabulary is 50304 instead of 50257:
```python
model = GPT(GPTConfig(vocab_size=50304))
```

**Why**: Rounding vocab size to multiples of 64 or 128 improves computational efficiency on modern GPUs.

## Conclusions

Implementing GPT-2 from scratch has been an incredibly formative experience. I understood:

1. **The importance of architecture**: Every component (attention, MLP, residuals, normalization) has a specific and critical role.

2. **Training is as important as architecture**: Optimization, learning rate scheduling, mixed precision, and parallelization are essential to make these models work.

3. **Optimizations make the difference**: FlashAttention, TF32, torch.compile, and DDP drastically reduce training times.

4. **Data engineering is critical**: Tokenization, efficient data loading, and memory management are fundamental.

This project gave me a deep understanding of how modern LLMs work and I now feel much more prepared to work with large language models.

## Resources That Helped Me

- Andrej Karpathy's video on GPT-2 from scratch
- Paper "Attention is All You Need" (Vaswani et al.)
- Paper GPT-2 "Language Models are Unsupervised Multitask Learners"
- PyTorch documentation on DDP and optimizations
- FlashAttention paper to understand kernel-level optimizations

## Next Steps

Now that I've implemented GPT-2, I'd like to:
1. Experiment with different datasets
2. Implement more advanced techniques (RoPE, GQA, Flash Attention 2)
3. Try more modern architectures (LLaMA, GPT-3 style)
4. Fine-tune on specific tasks
5. Implement RLHF (Reinforcement Learning from Human Feedback)

---

**Project completion date**: November 2025  
**Total study and implementation time**: ~3 weeks  
**Lines of code written**: ~450  
**Lessons learned**: Infinite 🚀
