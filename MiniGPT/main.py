import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import math
import tiktoken
import json
import os
import matplotlib.pyplot as plt
torch.manual_seed(1042)


class GPTConfig:
    block_size: int = 512  # max sequence length
    batch_size: int = 12
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    hidden_dim: int = n_embd
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257  # GPT-2's vocabulary size


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super(SingleHeadAttention, self).__init__()
        self.head_size = config.head_size
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.register_buffer("attention_mask", torch.tril(
            torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # seq_length 而非 block_size
        # 因为在 forward 时，输入的 x 可能小于 block_size
        batch_size, seq_length, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_length, :seq_length] == 0, float('-inf')) / math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.linear2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config)
                                    for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 现在的 SLM 模型，使用 tie weight 来减少参数
        # head: n_embd -> vocab_size, weight: n_embd x vocab_size
        self.tok_embeddings.weight = self.head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 是输入 token idx
        # targets 是目标 token idx
        batch_size, seq_length = idx.size()
        tok_embeddings = self.tok_embeddings(idx)
        pos_embeddings = self.pos_embeddings(
            torch.arange(seq_length, device=idx.device))
        # 广播机制
        x = tok_embeddings + pos_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is not None:
            batch_size, seq_length, vocab_size = logits.size()
            # 计算损失
            logits = logits.view(batch_size * seq_length, vocab_size)
            targets = targets.view(batch_size * seq_length)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits

    def generate(self, idx, max_new_tokens):
        # idx (batch_size, seq_length) 是输入 token idx
        # max_new_tokens 是生成的最大 token 数量
        for _ in range(max_new_tokens):
            # 取最新的 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            # logits (batch_size, seq_length, vocab_size) 是输出 logits
            # 取最后一个时间步的 logits
            logits = logits[:, -1, :]
            # 计算整个词汇表的概率分布
            probs = F.softmax(logits, dim=-1)
            # 采样一个 token，相比于取最大值，采样会让生成结果更具多样性
            # 也可以使用 temperature 采样来控制多样性
            idx_next = torch.multinomial(probs, num_samples=1)
            # (batch_size, seq_length + 1) 将采样的 token 添加到 idx 中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.encoded_data = []
        # 使用 <|endoftext|> 作为结束符
        self.eos_token = self.enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        raw_data = []
        with open(data, 'r', encoding='utf-8') as f:
            for line in f:
                texts = json.loads(line.strip())
                raw_data.append(texts)

        full_encoded = []
        for text in raw_data:
            encoded = self.enc.encode(text)
            full_encoded.extend(encoded + [self.eos_token])
        # 长文本截断
        for i in range(0, len(full_encoded), self.block_size):
            # 取 513 维，前 512 维是输入，后 512 维是目标
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * \
                    (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, idx):
        return self.enc.decode(idx)


def train(model, optimizer, scheduler, device, train_loader, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        return total_loss


def val(model, device, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    return total_loss


def main(epochs):
    config = GPTConfig()
    model = GPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / (2**20):.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)
    dataset = CustomDataset(
        "MiniGPT/dataset/mobvoi_seq_monkey_general_open_corpus_1000.json", config.block_size)
    print(f"Dataset size: {len(dataset)}")
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, scheduler,
                           device, train_loader, epoch)
        val_loss = val(model, device, val_loader)
        print(
            f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss / len(val_loader),
        }
        if not os.path.exists("MiniGPT/checkpoints"):
            os.makedirs("MiniGPT/checkpoints")
        if epoch % 20 == 0:
            torch.save(checkpoint, f"MiniGPT/checkpoints/epoch_{epoch}.pth")
        # plot loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("MiniGPT/loss_curve.png")


def inference(text):
    config = GPTConfig()
    model = GPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    enc = tiktoken.get_encoding("gpt2")
    checkpoint = torch.load(
        "MiniGPT/checkpoints/epoch_100.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def text_tensor(text, enc, config, device):
        tokens = enc.encode(text)
        eos_token = enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"})
        full_encoded = tokens + eos_token
        encoded_data = []
        for i in range(0, config.block_size):
            # 取 513 维，前 512 维是输入，后 512 维是目标
            chunk = full_encoded[i:i + config.block_size + 1]
            if len(chunk) < config.block_size + 1:
                chunk = chunk + eos_token * \
                    (config.block_size + 1 - len(chunk))
            encoded_data.append(chunk)
        return torch.tensor(encoded_data, dtype=torch.long).to(device)

    idx = text_tensor(text, enc, config, device)
    out = model.generate(idx, config.block_size)
    output = enc.decode(out[:, config.block_size:].data.to("cpu").numpy()[0])
    print("output:", output)


if __name__ == "__main__":
    # main(100)
    inference("Hello, world!")
