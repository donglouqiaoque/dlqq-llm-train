# -*- coding: utf-8 -*-
r"""模型定义（transformers 版）
=================================================================
训练（code\train.py）、对话（code\chat.py）、桥接服务（code\http_server.py）
都从这里 import，**不要再各自复制一份**。

实现从手写 GPT 换成了 HuggingFace transformers 的 GPT2LMHeadModel：
    · 仍然是 Decoder-only、字符级、从零训练；
    · 仍然用项目自己的 chars / stoi / itos，不引入 HuggingFace tokenizer；
    · 保存为 HuggingFace 标准目录：
          model/<名字>/config.json
          model/<名字>/model.safetensors（或 pytorch_model.bin）
          model/<名字>/chars.json
          model/<名字>/meta.json
      续训使用 Trainer 风格目录：
          model/<名字>/checkpoint-<step>/
              config.json / model.safetensors
              optimizer.pt / scheduler.pt / trainer_state.json / rng_state.pth
      旧版自定义 .pt 不能直接加载，需要重新训练。

接口保持和旧版一致：
    GPT(vocab, block_size, n_layer, n_head, n_embd, dropout, grad_checkpoint=False)
    forward(idx)             -> logits
    forward(idx, targets)    -> (logits, loss)
    sample(idx, n, ...)      -> 逐 token 产出 id（流式）
    generate(idx, n, ...)    -> 完整序列（内部用 HF generate）
"""
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

# HuggingFace 标准目录里的文件名；chars / meta 是本项目额外元数据
CHARS_FILE = "chars.json"
META_FILE = "meta.json"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def has_weights(path):
    """目录是否是本项目可加载的 HF 标准权重目录。"""
    return (os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "config.json"))
            and any(os.path.isfile(os.path.join(path, w)) for w in WEIGHT_FILES))


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


class GPT(nn.Module):
    """对 GPT2LMHeadModel 的薄封装：保持旧版调用接口，内部全部交给 transformers。"""

    def __init__(self, vocab, block_size, n_layer, n_head, n_embd, dropout,
                 grad_checkpoint=False):
        super().__init__()
        self.block_size = block_size
        self.grad_checkpoint = grad_checkpoint
        cfg = GPT2Config(
            vocab_size=vocab,
            n_positions=block_size,
            n_ctx=block_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * n_embd,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            tie_word_embeddings=True,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
        self.net = GPT2LMHeadModel(cfg)
        # 梯度检查点和 KV cache 不能同时开；训练时关掉 cache
        self.net.config.use_cache = not grad_checkpoint
        if grad_checkpoint:
            self.net.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, load_dir):
        """加载本项目自己训练出的 HF 标准目录（不是第三方预训练权重）。"""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.net = GPT2LMHeadModel.from_pretrained(load_dir)
        obj.block_size = int(obj.net.config.n_positions)
        obj.grad_checkpoint = False
        obj.net.config.use_cache = True
        return obj

    def save_pretrained(self, save_dir):
        """按 HuggingFace 标准目录保存 config.json + model.safetensors。"""
        os.makedirs(save_dir, exist_ok=True)
        try:
            self.net.save_pretrained(save_dir, safe_serialization=True)
        except (ImportError, RuntimeError):
            # 没有 safetensors 时退回 pytorch_model.bin，目录格式仍然标准
            self.net.save_pretrained(save_dir, safe_serialization=False)

    def load_weights(self, load_dir):
        """把标准目录里的权重加载到当前模型（保留当前设备）。"""
        other = GPT2LMHeadModel.from_pretrained(load_dir)
        self.net.load_state_dict(other.state_dict())
        return self

    def forward(self, idx, targets=None):
        # 训练数据的 targets 已经是「下一个字符」（train.py 的 y），
        # 不能再交给 HF 的 labels 做一次 shift，否则会错位一位。
        # attention_mask 全 1：字符级输入没有 padding，显式传可以避免 HF 的警告。
        attention_mask = torch.ones_like(idx)
        out = self.net(input_ids=idx, attention_mask=attention_mask)
        if targets is None:
            return out.logits
        loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)),
                               targets.reshape(-1))
        return out.logits, loss

    @torch.no_grad()
    def sample(self, idx, n, temperature=1.0, top_k=None):
        """逐 token 产出 id；对话和 HTTP 服务用它做流式输出。"""
        for _ in range(n):
            cond = idx[:, -self.block_size:]
            logits = self.net(input_ids=cond,
                              attention_mask=torch.ones_like(cond)).logits[:, -1, :]
            logits = logits / max(float(temperature), 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            yield int(nxt.item())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """整段生成。

        不能直接用 HF 的 generate：GPT-2 的位置表只有 block_size 长，
        先给长 prompt、再生成超过 block_size 个 token 会越界。
        这里复用 sample 的逐步截断逻辑，保证上下文永远不超过 block_size。
        """
        out = idx
        for tid in self.sample(idx, max_new_tokens, temperature, top_k):
            nxt = torch.tensor([[tid]], dtype=out.dtype, device=out.device)
            out = torch.cat([out, nxt], dim=1)
        return out
