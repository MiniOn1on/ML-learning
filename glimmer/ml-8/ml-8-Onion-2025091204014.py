import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

## ====================== 【新增：显存打印工具函数】 ======================
def print_gpu_mem(tag: str):
    """显存实测口径：PyTorch tensor 分配显存（不含驱动/上下文开销）
    tag: 标记是哪个阶段的显存快照
    """
    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU MEM] {tag:20s} | allocated: {alloc_mb:.2f} MB | peak allocated: {peak_alloc_mb:.2f} MB | reserved: {reserved_mb:.2f} MB")
## =====================================================================

## 超参设置（每个参数都有讲究！）
batch_size = 16     # 批处理大小
block_size = 128     # 上下文长度
max_iters = 5000    # 训练轮数
learning_rate = 1e-3 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64         # embedding维度/d
n_head = 4          # 多头注意力头数
n_layer = 4         # transformer层数
dropout = 0.0       # dropout率
torch.manual_seed(42) # 随机种子，保证可复现

# 读取莎士比亚文本
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"数据集总字符数: {len(text):,}")
print(f"前300个字符预览:\n{text[:300]}")

# 构建词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词汇表: {''.join(chars)}")
print(f"词汇表大小: {vocab_size}")

# 简单tokenizer实现
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 测试编解码
print(f"编码测试: {encode('Hello World!')}")
print(f"解码测试: {decode(encode('Hello World!'))}")

def get_batch(split):
    """
    生成训练批次数据
    你需要实现：
    1. 从训练/验证集中随机采样
    2. 生成输入序列x和目标序列y
    3. 返回正确维度的tensor
    """
    ###################
    # 你的代码在这里
    # 1. 将文本编码为 Tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # 2. 划分训练集和验证集（90% 训练，10% 验证）
    n = int(0.9 * len(data))
    data = data[:n] if split == 'train' else data[n:]

    # 3. 随机采样起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # 4. 构造输入 x 和目标 y
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    # 5. 移动到指定设备
    x, y = x.to(device), y.to(device)

    return x, y
    ###################

class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, head_size):
        super().__init__()
        # QKV投影矩阵
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角mask矩阵（防止看到未来）
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        ###################
        # 注意力计算逻辑
        # 1. 计算Q, K, V
        # 2. 计算注意力分数
        # 3. 应用mask
        # 4. 加权求和
        ###################
        # 1. 计算 Q、K、V
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # 2. 计算注意力分数，并进行缩放
        A = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # A: (B, T, T)
        # 3. 应用下三角 mask，防止看到未来的信息
        A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 4. Softmax 转换为注意力权重
        A = F.softmax(A, dim=-1)
        # 5. 根据注意力权重对 V 加权求和
        out = A @ v
        # out: (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """多头注意力：并行处理多个注意力头"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ###################
        # 多头并行计算与合并
        ###################
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 投影 + dropout
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """前馈神经网络：简单而强大"""
    ###################
    # 实现两层线性变换+激活
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    ###################

class Block(nn.Module):
    """Transformer Block：communication + computation"""
    ###################
    # 组装完整的transformer块
    # LayerNorm + MultiHeadAttention + LayerNorm + FeedForward
    ###################
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """完整的语言模型"""

    def __init__(self):
        super().__init__()
        # embedding表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 堆叠transformer块
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])#n_layer层transformer block
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        ###################
        # 完整前向传播
        # 1. token + position embedding
        # 2. 通过transformer块
        # 3. 输出预测
        # 4. 计算损失（如果有target）
        ###################
        B, T = idx.shape
        # 1. token embedding + position embedding
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,n_embd)
        x = tok_emb + pos_emb  #广播相加，总输入
        # 2. 经过堆叠的Transformer Block
        x = self.blocks(x)  # (B,T,n_embd)
        x = self.ln_f(x)  # 最后的层归一化
        # 3. 预测logits
        logits = self.lm_head(x)  # (B,T,vocab_size)
        # 4. 计算损失（只有传入target才计算loss，生成阶段target=None）
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """文本生成：让AI说话！"""
        for _ in range(max_new_tokens):
            # 裁剪到block_size
            idx_cond = idx[:, -block_size:]
            # 预测下一个token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只要最后一个时间步
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

@torch.no_grad()
def estimate_loss():
    """评估训练和验证损失"""
    out = {}
    model.eval()
    ###################
    # 计算平均损失
    eval_iters = 200  # 200样本求平均
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    ###################
    model.train()
    return out

# 实例化模型
model = BigramLanguageModel().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
## ====================== 【新增：模型加载完成后，打印初始显存】 ======================
print_gpu_mem("model loaded, before train")
## ==================================================================================

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_losses, val_losses = [], []

for iter in range(max_iters):
    ###################
    # 训练循环实现
    # 1. 获取batch
    # 2. 前向传播
    # 3. 计算损失
    # 4. 反向传播
    # 5. 参数更新
    # 6. 定期评估
    ###################
    # 1. 获取一个batch训练数据
    xb, yb = get_batch('train')

    # 2. 前向传播，3. 得到loss
    logits, loss = model(xb, yb)

    # 4. 反向传播
    optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度
    loss.backward()  # 计算梯度

    # 5. 参数更新
    optimizer.step()

    # 6. 定期评估损失，保存画图数据
    if iter % 200 == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        print(f"iter {iter:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
        ## ====================== 【新增：评估时打印显存快照】 ======================
        print_gpu_mem(f"iter {iter} eval")
        ## ========================================================================

## ====================== 【新增：训练结束，全局峰值显存】 ======================
if torch.cuda.is_available():
    peak_total = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"\n[FINAL PEAK] 训练全程最大张量显存: {peak_total:.2f} MB")
## ===========================================================================

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='#FF6B6B')
plt.plot(val_losses, label='Validation Loss', color='#4ECDC4')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Progress: Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

# 生成莎士比亚风格文本
print("AI莎士比亚开始创作...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated_text)
