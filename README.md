# GPT-2 from Scratch

Complete implementation of the GPT-2 model from scratch in PyTorch, following best practices and modern optimization techniques.

## Project Structure

```
gpt2/
├── gpt2_trainer.py      # Model implementation and training loop
├── training_data.txt    # Training dataset
├── APPRENDIMENTI.md     # Detailed documentation of my learning journey
├── .gitignore          # Files to ignore in git
└── README.md           # This file
```

## Key Features

- **Complete GPT-2 Architecture**: Self-Attention, MLP blocks, LayerNorm, Residual connections
- **Modern Optimizations**: 
  - FlashAttention (scaled_dot_product_attention)
  - Mixed Precision Training (bfloat16)
  - Gradient Accumulation
  - Model Compilation (torch.compile)
  - TF32 support
- **Distributed Training**: Full support for Distributed Data Parallel (DDP)
- **Learning Rate Scheduling**: Cosine decay with linear warmup
- **Gradient Clipping**: Training stabilization
- **Weight Tying**: Weight sharing between embeddings and output layer

## Requirements

```bash
pip install torch tiktoken transformers
```

## Usage

### Single GPU/CPU Training

```bash
python gpt2_trainer.py
```

### Distributed Training (multi-GPU)

```bash
torchrun --standalone --nproc_per_node=8 gpt2_trainer.py
```

## Model Configuration

| Parameter | Value (GPT-2 base) |
|-----------|---------------------|
| n_layers  | 12                  |
| n_heads   | 12                  |
| n_embd    | 768                 |
| block_size| 1024                |
| vocab_size| 50304 (padded)      |
| Parameters| ~124M               |

## Hyperparameters

- **Total Batch Size**: 524,288 tokens
- **Micro Batch Size**: 16
- **Sequence Length**: 1024
- **Max Learning Rate**: 6e-4
- **Min Learning Rate**: 6e-5
- **Weight Decay**: 0.1
- **Warmup Steps**: 10
- **Max Steps**: 50

## Loading Pre-trained Weights

The code supports loading weights from HuggingFace:

```python
model = GPT.from_pretrained('gpt2')  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
```

## Documentation

For an in-depth understanding of the implementation and concepts, see [APPRENDIMENTI.md](APPRENDIMENTI.md).

## References

- Paper "Attention is All You Need" (Vaswani et al., 2017)
- Paper "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- Andrej Karpathy's GPT-2 video tutorial
- PyTorch Documentation

## Notes

This project was created for educational purposes to deeply understand how Large Language Models work.

## License

MIT License
