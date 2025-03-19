# SimpleLLM: A Lightweight Transformer-based Language Model

## Overview

SimpleLLM is a lightweight transformer-based language model designed to train on the Wikipedia dataset. The model is implemented using PyTorch and the Hugging Face Transformers library. This project serves as an educational tool for understanding how small-scale language models are trained.

## Features

- Uses a simple transformer architecture.

- Trains on the Wikipedia dataset.

- Implements tokenization using a pre-trained GPT-2 tokenizer.

- Generates text samples during training to evaluate performance.

- Efficient training with positional embedding fixes and memory management.

## Installation

- Ensure you have Python installed (preferably version 3.8 or later). Then, install the required dependencies:

** pip install torch transformers datasets tqdm ** 

- Running the Training Script

- To train the model, run:

** python simllm.py ** 

## Fixes and Improvements

- Several key changes were made to resolve previous errors and enhance performance:

- Position Embedding Fix: Added a modulo operation to ensure positions never exceed the maximum length:
  ** positions = positions % self.max_length ** 

- positions = positions % self.max_length

- Consistency in Max Length: Introduced self.max_length in both the Transformer and SimpleLLM classes to ensure uniform sequence length handling.

- Reduced Generation Frequency: Sample text generation now occurs every 5 epochs (or on the final epoch) to improve training efficiency.

- Context Management: Improved the generate method to ensure context length remains within the model's limit.

## Issue Resolution

The error occurred because the positions tensor exceeded the maximum sequence length in the position embedding layer. This fix ensures proper indexing and prevents out-of-range errors.

## Sample Output

After training, the model generates text based on a given prompt. Example:

** The quick brown fox jumps over the lazy dog. The... ** 

## Future Improvements

- Implement attention masking for better handling of longer sequences.

- Fine-tune on domain-specific datasets for improved results.

- Experiment with different transformer architectures for optimization.



License:
MIT 
