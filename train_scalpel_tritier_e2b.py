from collections import Counter
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ----------------------------------------------------------------------
# 1. 对抗数据集加载 (50/50 文理对比语料)
# ----------------------------------------------------------------------
class ScalpelDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = [{
            "role": "user",
            "content": item["prompt"]
        }, {
            "role": "assistant",
            "content": item["response"]
        }]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        big_target = int(item["big_target"])
        little_group = int(item["little_group"])

        # 32 微专家映射
        if big_target == 1 and little_group < 16:
            little_target = little_group + 16
        else:
            little_target = little_group

        little_target = little_target % 32

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "big_target": torch.tensor(big_target, dtype=torch.long),
            "little_target": torch.tensor(little_target, dtype=torch.long),
        }


# ----------------------------------------------------------------------
# 2. 通用 LoRA 适配单元
# ----------------------------------------------------------------------
class ScalpelLoRAModule(nn.Module):

    def __init__(
        self,
        hidden_dim,
        rank=16,
        lora_alpha=16.0,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim,
                                rank,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(rank,
                                hidden_dim,
                                bias=False,
                                device=device,
                                dtype=dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 3. 大中小三层包装器 (原生 BF16 高速版)
# ----------------------------------------------------------------------
class ScalpelTriTierWrapper(nn.Module):

    def __init__(
        self,
        original_mlp,
        hidden_dim,
        sci_rank=64,  # 已调整为 64 (显存计算减半，吞吐暴增)
        micro_rank=16,  # 32 微专家保持 Rank-16
        num_experts=32,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_experts = num_experts

        # 【大核】文科原生底座 (原生 BF16，完全锁死)
        self.big_arts = original_mlp
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # 【中核】理科大核：Rank-64 Dense LoRA
        self.sci_dense_lora = ScalpelLoRAModule(
            hidden_dim=hidden_dim,
            rank=sci_rank,
            lora_alpha=float(sci_rank * 2),
            device=device,
            dtype=dtype,
        )

        # 文理大核路由器 (hidden_dim -> 2)
        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.last_router_big_logits = None

        # 【小核】32 微专家池 + 调度路由器 (hidden_dim -> 32)
        self.router_little = nn.Linear(hidden_dim,
                                       num_experts,
                                       bias=False,
                                       device=device,
                                       dtype=dtype)
        self.last_router_little_logits = None

        self.lora_pool = nn.ModuleList([
            ScalpelLoRAModule(
                hidden_dim=hidden_dim,
                rank=micro_rank,
                lora_alpha=float(micro_rank * 1),
                device=device,
                dtype=dtype,
            ) for _ in range(num_experts)
        ])
        self.current_little_target = None

    def forward(self, x):
        # 1. 大核输出 (原生 BF16 计算极速)
        arts_out = self.big_arts(x)

        # 2. 中核理科增量
        sci_out = arts_out + self.sci_dense_lora(x)

        # 3. 大中核软门控
        router_big_logits = self.router_big(x)
        self.last_router_big_logits = router_big_logits
        weights_big = torch.softmax(router_big_logits, dim=-1)

        big_tier_out = (weights_big[..., 0:1] *
                        arts_out) + (weights_big[..., 1:2] * sci_out)

        # 4. 小核微专家路由
        router_little_logits = self.router_little(x)
        self.last_router_little_logits = router_little_logits

        batch_size = x.size(0)
        micro_outs = []
        for b in range(batch_size):
            exp_idx = self.current_little_target[b].item()
            micro_outs.append(self.lora_pool[exp_idx](x[b:b + 1]))
        micro_out = torch.cat(micro_outs, dim=0)

        # 5. 三层融合输出
        return big_tier_out + 0.3 * micro_out


# ----------------------------------------------------------------------
# 4. 解码器层自适应定位
# ----------------------------------------------------------------------
def locate_gemma_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers

    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    for name, module in model.named_modules():
        if name.endswith("language_model.layers") or name.endswith(
                "model.layers"):
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return module

    raise RuntimeError("无法定位模型解码器层，请检查模型结构！")


# ----------------------------------------------------------------------
# 5. 主训练流水线
# ----------------------------------------------------------------------
def main():
    model_id = "aifeifei798/Heretic-Scalpel-E2B"
    data_path = "dual_contrast_data.jsonl"

    assert os.path.exists(data_path), f"找不到训练数据 {data_path}！"

    print("=" * 75)
    print("⚡ 启动【Gemma 4 E2B】原生 BF16 极速三层大小核特训 (Rank-64 中核)")
    print("=" * 75)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 彻底移除 4-bit 量化，使用原生 BF16 无损极速加载
    print(f"[*] 正在载入 E2B 原生基座 (纯 BF16 全速计算)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)

    # 卸载多模态组件
    for attr in [
            "vision_tower",
            "audio_tower",
            "visual",
            "audio",
            "mm_projector",
    ]:
        for parent in [model, getattr(model, "model", None)]:
            if parent is not None and hasattr(parent, attr):
                module = getattr(parent, attr)
                if module is not None:
                    module.to("cpu")
    torch.cuda.empty_cache()

    # 冻结原生基座
    for param in model.parameters():
        param.requires_grad = False

    layers = locate_gemma_layers(model)
    num_layers = len(layers)

    text_cfg = getattr(model.config, "text_config", model.config)
    hidden_dim = getattr(text_cfg, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = layers[0].mlp.gate_proj.in_features

    print(f"[+] 捕获 E2B 解码主干：{num_layers} 层，Hidden Dim: {hidden_dim}")
    print(f"[*] 部署三层架构：中核 (Dense LoRA R=64) + 小核 (32 微专家 R=16)...")

    # 2. 挂载三层结构 (sci_rank 改为 64)
    for layer in layers:
        layer.mlp = ScalpelTriTierWrapper(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            sci_rank=64,  # 中核容量从 128 改为 64
            micro_rank=16,
            num_experts=32,
            device="cuda:0",
            dtype=dtype,
        )

    model.gradient_checkpointing_enable()

    # 收集参数
    sci_params = []
    router_params = []
    micro_params = []

    for layer in layers:
        sci_params.extend([
            p for p in layer.mlp.sci_dense_lora.parameters() if p.requires_grad
        ])
        router_params.extend(
            [p for p in layer.mlp.router_big.parameters() if p.requires_grad])
        router_params.extend([
            p for p in layer.mlp.router_little.parameters() if p.requires_grad
        ])
        micro_params.extend(
            [p for p in layer.mlp.lora_pool.parameters() if p.requires_grad])

    total_trainable = sum(p.numel()
                          for p in (sci_params + router_params + micro_params))
    print(f"\n[✔] 架构装配完成！参数账单：")
    print(f"    - 原生 BF16 大核基座: ~4.5 GB")
    print(
        f"    - 中核 (理科 Dense LoRA R=64): {sum(p.numel() for p in sci_params)/1e6:.2f} M"
    )
    print(f"    - 小核 (微专家池): {sum(p.numel() for p in micro_params)/1e6:.2f} M")
    print(f"    - 双级路由器: {sum(p.numel() for p in router_params)/1e6:.2f} M")
    print(f"    - 总可训练参数量: {total_trainable/1e6:.2f} M")

    # 3. 提升 Micro Batch 填满 GPU 核心
    MICRO_BATCH = 4  # 从 1 改为 4，吞吐翻倍
    GRAD_ACCUM = 4  # 4 * 4 = 16 (等效 Batch Size 保持 16 不变)
    MAX_LENGTH = 512

    dataset = ScalpelDataset(data_path, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        shuffle=True,
        num_workers=4,  # 加大 worker 预加载
        pin_memory=True,
    )

    # 4. 换用 CUDA 原生 Fused AdamW 优化器，比 8-bit 快得多
    optimizer = torch.optim.AdamW(
        [
            {
                "params": sci_params,
                "lr": 5e-5,
                "weight_decay": 0.01
            },
            {
                "params": router_params,
                "lr": 1e-4,
                "weight_decay": 0.01
            },
            {
                "params": micro_params,
                "lr": 1e-4,
                "weight_decay": 0.01
            },
        ],
        fused=True,
    )
    print("[✔] 挂载原生 PyTorch CUDA Fused AdamW 优化器！")

    criterion_ce = nn.CrossEntropyLoss()
    total_steps = len(dataloader) // GRAD_ACCUM
    print(f"\n[+] 开始极速三层大小核微调训练 (总 Step 数: {total_steps})...")

    model.train()
    start_time = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0",
                                                    non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        big_target = batch["big_target"].to("cuda:0", non_blocking=True)
        little_target = batch["little_target"].to("cuda:0", non_blocking=True)

        for layer in layers:
            layer.mlp.current_little_target = little_target

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        lm_loss = outputs.loss

        # 路由器监督
        mask = attention_mask == 1
        target_big_seq = big_target.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]
        target_little_seq = little_target.unsqueeze(1).expand(-1,
                                                              MAX_LENGTH)[mask]

        big_router_loss = 0.0
        little_router_loss = 0.0

        for layer in layers:
            active_logits_big = layer.mlp.last_router_big_logits[mask]
            active_logits_little = layer.mlp.last_router_little_logits[mask]

            big_router_loss += criterion_ce(active_logits_big, target_big_seq)
            little_router_loss += criterion_ce(active_logits_little,
                                               target_little_seq)

        total_loss = (lm_loss + 0.1 * (big_router_loss / num_layers) + 0.1 *
                      (little_router_loss / num_layers)) / GRAD_ACCUM

        total_loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM
            if global_step % 10 == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                current_loss = total_loss.item() * GRAD_ACCUM
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                mem_used = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} "
                    f"| LM: {lm_loss.item():.4f} | 显存峰值: {mem_used:.1f} GB | 速度: {speed:.1f} samples/s"
                )
            if global_step >= 50:
                print("\n[✔] 达到黄金拟合步数 (Step 50)，提前收工防过拟合！")
                break

    print(f"\n[✔] 极速特训完成！总耗时仅: {(time.time() - start_time)/60:.2f} 分钟")

    save_path = "scalpel_tritier_e2b_weights.pt"
    print(f"[*] 正在保存轻量适配器权重至 {save_path}...")

    state_to_save = {}
    for i, layer in enumerate(layers):
        state_to_save[f"layer_{i}_sci_dense_lora"] = (
            layer.mlp.sci_dense_lora.state_dict())
        state_to_save[f"layer_{i}_router_big"] = (
            layer.mlp.router_big.state_dict())
        state_to_save[f"layer_{i}_router_little"] = (
            layer.mlp.router_little.state_dict())
        state_to_save[f"layer_{i}_micro_loras"] = (
            layer.mlp.lora_pool.state_dict())

    torch.save(state_to_save, save_path)
    print(f"[✔] 权重成功保存为 {save_path}！")


if __name__ == "__main__":
    main()
