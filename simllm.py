import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Define a simple tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        unique_chars = sorted(set(''.join(texts)))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        return self
    
    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] if idx in self.idx_to_char else '' for idx in indices])

# Define a simple text dataset
class TextDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.encoded_text = tokenizer.encode(text)
        
    def __len__(self):
        return max(0, len(self.encoded_text) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        inputs = self.encoded_text[idx:idx+self.seq_length]
        targets = self.encoded_text[idx+1:idx+self.seq_length+1]
        return torch.tensor(inputs), torch.tensor(targets)

# Define a simple attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        
        # Split embedding into self.heads pieces
        query = query.reshape(batch_size, seq_length, self.heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.heads, self.head_dim)
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        # Compute attention scores
        energy = torch.einsum("bqhd,bkhd->bhqk", [query, key])
        # Mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)
        out = torch.einsum("bhql,blhd->bqhd", [attention, value]).reshape(
            batch_size, seq_length, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Define a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Define the transformer encoder
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.max_length = max_length
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape
        
        # Important fix: make sure positions don't exceed max_length
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        # Apply modulo to ensure positions are within range
        positions = positions % self.max_length
        
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        for layer in self.layers:
            out = layer(out, mask)
            
        out = self.fc_out(out)
        return out

# Define a simple language model
class SimpleLLM:
    def __init__(self, embed_size=64, num_layers=2, heads=2, device="cpu"):
        self.device = device
        self.tokenizer = SimpleTokenizer()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.model = None
        self.max_length = 512  # Default max sequence length
        
    def train(self, texts, batch_size=32, epochs=10, seq_length=16, lr=0.001):
        # Tokenize text
        self.tokenizer.fit(texts)
        self.max_length = seq_length
        
        # Initialize model
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            device=self.device,
            forward_expansion=4,
            dropout=0.1,
            max_length=seq_length,
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = TextDataset(''.join(texts), seq_length, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs.reshape(-1, self.tokenizer.vocab_size), targets.reshape(-1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
            
            # Generate sample text after each epoch
            if epoch % 5 == 0 or epoch == epochs - 1:
                print("Sample text:", self.generate("The ", max_length=50))
            
    def generate(self, prompt, max_length=100, temperature=1.0):
        if self.model is None:
            return "Model not trained yet."
        
        self.model.eval()
        with torch.no_grad():
            encoded_prompt = self.tokenizer.encode(prompt)
            context = torch.tensor(encoded_prompt).unsqueeze(0).to(self.device)
            
            generated = list(encoded_prompt)
            
            for _ in range(max_length - len(encoded_prompt)):
                if len(context[0]) > self.max_length:  # Keep context within max_length
                    context = context[:, -self.max_length:]
                
                outputs = self.model(context)
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                context = torch.cat([context, torch.tensor([[next_token]]).to(self.device)], dim=1)
                
            return self.tokenizer.decode(generated)

# Example usage
def train_simple_llm():
    # Sample training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Language models are trained to predict the next word in a sentence.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large amounts of data to train effectively.",
        "Neural networks consist of layers of interconnected nodes."
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize and train the model
    llm = SimpleLLM(embed_size=64, num_layers=2, heads=2, device=device)
    llm.train(texts, batch_size=8, epochs=20, seq_length=32)
    
    # Generate text
    prompt = "The model"
    generated_text = llm.generate(prompt, max_length=100, temperature=0.8)
    print(f"Generated text: {generated_text}")
    
    return llm

if __name__ == "__main__":
    train_simple_llm()