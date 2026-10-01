from collections import Counter
import os
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# 32 专家语义标签字典 (0~15 文科组, 16~31 理科组)
# ----------------------------------------------------------------------
EXPERT_NAMES = {
    # 0~15: 文科与人文社科微专家
    0: "Arts_Prose",
    1: "Arts_Poetry",
    2: "Arts_Fiction",
    3: "Arts_Drama",
    4: "Arts_Essay",
    5: "Arts_History",
    6: "Arts_Culture",
    7: "Arts_Linguistics",
    8: "Arts_Philosophy",
    9: "Arts_Ethics",
    10: "Arts_Rhetoric",
    11: "Arts_Translation",
    12: "Arts_Critique",
    13: "Arts_Dialogue",
    14: "Arts_Summary",
    15: "Arts_Chat",
    # 16~31: STEM 与理工代码微专家
    16: "Code_Algo",
    17: "Code_DS",
    18: "Code_Debug",
    19: "Code_Arch",
    20: "Code_Syntax",
    21: "Code_Optim",
    22: "Math_Algebra",
    23: "Math_Geo",
    24: "Math_Prob",
    25: "Math_Arith",
    26: "Math_Calculus",
    27: "Math_Logic",
    28: "Sci_Physics",
    29: "Sci_Chem",
    30: "Sci_Biology",
    31: "Sci_General",
}


# ----------------------------------------------------------------------
# 全局三层全息遥测监控探针
# ----------------------------------------------------------------------
class TriTierTelemetryMonitor:

    def __init__(self):
        self.reset()

    def reset(self):
        self.arts_weights_sum = 0.0
        self.sci_weights_sum = 0.0
        self.total_tokens_routed = 0
        self.expert_activation_counter = Counter()

    def record_step(self, weights_big, chosen_experts):
        self.arts_weights_sum += weights_big[..., 0].sum().item()
        self.sci_weights_sum += weights_big[..., 1].sum().item()
        self.total_tokens_routed += weights_big.shape[0] * weights_big.shape[1]
        flat_experts = chosen_experts.view(-1).tolist()
        self.expert_activation_counter.update(flat_experts)


GLOBAL_TELEMETRY = TriTierTelemetryMonitor()


# ----------------------------------------------------------------------
# 1. 通用 LoRA 适配单元
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

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 2. 推理专用【大·中·小】三层架构包装器 (原生 BF16 + R=64 中核)
# ----------------------------------------------------------------------
class ScalpelTriTierInferenceWrapper(nn.Module):

    def __init__(
        self,
        original_mlp,
        hidden_dim,
        sci_rank=64,  # 与训练脚本的 Rank-64 严格对齐
        micro_rank=16,
        num_experts=32,
        residual_scale=0.08,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_experts = num_experts
        self.residual_scale = residual_scale

        # 【大核】文科原生底座 (BF16 只读锁死)
        self.big_arts = original_mlp

        # 【中核】理科大核 (Dense LoRA R=64)
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

        # 【小核】32 微专家池 + 调度器 (hidden_dim -> 32)
        self.router_little = nn.Linear(hidden_dim,
                                       num_experts,
                                       bias=False,
                                       device=device,
                                       dtype=dtype)
        self.lora_pool = nn.ModuleList([
            ScalpelLoRAModule(
                hidden_dim=hidden_dim,
                rank=micro_rank,
                lora_alpha=float(micro_rank * 1),
                device=device,
                dtype=dtype,
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # 1. 大核原生底座输出
        arts_out = self.big_arts(x)

        # 2. 中核理科增量输出
        sci_out = arts_out + 0.1 * self.sci_dense_lora(x)

        # 3. 大中核软门控加权
        router_big_logits = self.router_big(x)
        weights_big = torch.softmax(router_big_logits, dim=-1)

        big_tier_out = (weights_big[..., 0:1] *
                        arts_out) + (weights_big[..., 1:2] * sci_out)

        # 4. 小核微专家路由选择
        router_little_logits = self.router_little(x)
        chosen_experts = torch.argmax(router_little_logits, dim=-1)

        # 全局战报记录
        GLOBAL_TELEMETRY.record_step(weights_big, chosen_experts)

        # 选出主导专家做残差注入
        top1_expert_id = chosen_experts[0, -1].item()
        micro_out = self.lora_pool[top1_expert_id](x)

        # 5. 三层融合输出
        return big_tier_out + 0.02 * micro_out


# ----------------------------------------------------------------------
# 3. 辅助函数：定位 Gemma 4 解码层
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
    raise RuntimeError("无法定位解码器层，请检查模型结构！")


# ----------------------------------------------------------------------
# 4. 生成推理与全息透视战报输出
# ----------------------------------------------------------------------
def generate_and_diagnose(
    model,
    tokenizer,
    prompt,
    system_prompt="You are a helpful and versatile assistant.",
    max_new_tokens=2048,
):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            }]
        },
    ]

    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print(f"\n👤 提问: {prompt}\n")

    # 精准终止符
    stop_tokens = ["<end_of_turn>", "<eos>", "<|endoftext|>"]
    stop_ids = [tokenizer.eos_token_id]
    for st in stop_tokens:
        tid = tokenizer.convert_tokens_to_ids(st)
        if tid is not None and isinstance(tid, int):
            stop_ids.append(tid)

    GLOBAL_TELEMETRY.reset()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,  # 稍微压制重复
            no_repeat_ngram_size=4,  # 【关键】禁止任何 4 词以上的机械复读！
            eos_token_id=list(set(stop_ids)),
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    num_tokens = len(generated_tokens)
    speed = (num_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"🤖 回答: {response.strip()}\n")
    print(
        f"⚡ 极速推理: {speed:.1f} tokens/s ({num_tokens} tokens, 耗时: {elapsed_ms:.0f} ms)"
    )

    # 统计能耗百分比
    tot = (GLOBAL_TELEMETRY.arts_weights_sum +
           GLOBAL_TELEMETRY.sci_weights_sum) or 1.0
    arts_pct = (GLOBAL_TELEMETRY.arts_weights_sum / tot) * 100.0
    sci_pct = (GLOBAL_TELEMETRY.sci_weights_sum / tot) * 100.0

    def make_bar(pct, length=22):
        filled = int(round((pct / 100.0) * length))
        return "█" * filled + " " * (length - filled)

    # 打印全息诊断雷达
    print("═" * 72)
    print("🌌【E2B 大中小三层蜂群架构 · 全息透视战报】:")
    print(f"   🏛️  大核 (文科只读底座):    {arts_pct:5.1f}% [{make_bar(arts_pct)}]")
    print(f"   🔬 中核 (理科 Dense LoRA): {sci_pct:5.1f}% [{make_bar(sci_pct)}]")
    print("─" * 72)
    print("🪐【小核微专家命中热度榜 (Top 5 Active Micro-Experts)】:")

    top_active = GLOBAL_TELEMETRY.expert_activation_counter.most_common(5)
    if top_active:
        for exp_id, count in top_active:
            name = EXPERT_NAMES.get(exp_id, f"Expert_{exp_id}")
            group_tag = "文科组" if exp_id < 16 else "理科组"
            print(
                f"   ✨ #{exp_id:02d} [{group_tag} | {name:<16}]: {count:>6,d} 次路由调用"
            )
    else:
        print("   (未捕获微专家调用)")
    print("═" * 72)


# ----------------------------------------------------------------------
# 5. 主测试执行入口
# ----------------------------------------------------------------------
def main():
    model_id = "aifeifei798/Heretic-Scalpel-E2B"
    weights_path = "scalpel_tritier_e2b_weights.pt"

    if not os.path.exists(weights_path):
        print(f"❌ 错误: 找不到权重文件 {weights_path}！请确认训练是否完成。")
        return

    print("=" * 72)
    print("🚀 载入 E2B 大中小三层蜂群推理引擎 (原生 BF16 极速模式)")
    print("=" * 72)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] 正在加载原生 BF16 底座...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )

    # 启用 KV 缓存加速自回归生成
    model.config.use_cache = True

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
                setattr(parent, attr, None)
    torch.cuda.empty_cache()

    layers = locate_gemma_layers(model)
    num_layers = len(layers)

    text_cfg = getattr(model.config, "text_config", model.config)
    hidden_dim = getattr(text_cfg, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = layers[0].mlp.gate_proj.in_features

    print(f"[*] 挂载三层蜂群包装器 (Hidden: {hidden_dim}, sci_rank: 64)...")
    for layer in layers:
        layer.mlp = ScalpelTriTierInferenceWrapper(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            sci_rank=64,  # 与训练脚本 Rank-64 对齐
            micro_rank=16,
            num_experts=32,
            residual_scale=0.3,
            device="cuda:0",
            dtype=dtype,
        )

    # 注入刚出炉的微调器官权重
    print(f"[*] 正在注入特训器官权重: {weights_path} ...")
    weights = torch.load(weights_path, map_location="cuda:0")
    for i, layer in enumerate(layers):
        layer.mlp.sci_dense_lora.load_state_dict(
            weights[f"layer_{i}_sci_dense_lora"])
        layer.mlp.router_big.load_state_dict(weights[f"layer_{i}_router_big"])
        layer.mlp.router_little.load_state_dict(
            weights[f"layer_{i}_router_little"])
        layer.mlp.lora_pool.load_state_dict(weights[f"layer_{i}_micro_loras"])

    model.eval()

    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[✔] 蜂群推理引擎就绪！当前显存仅占用: {vram_peak:.2f} GB\n")

    # 自动化对比基准用例
    test_cases = [
        "写一首七言绝句，描写秋夜独坐窗前听雨的孤寂心境，讲究平仄和诗意境象。",
        "用 Python 写一个支持泛型的并发优先队列，并严格证明它的时间复杂度；再解释一下为什么在多线程环境下需要使用 CAS 原语。",
        "为什么天空是蓝色的，而夕阳是红色的？请用清晰的瑞利散射物理机制解释。",
    ]

    # print("🚀 [Phase 1: 预置文理跨域对照评测]\n")
    # for prompt in test_cases:
    #     generate_and_diagnose(model, tokenizer, prompt)

    print("\n🚀 [Phase 2: 交互式对话体验] (输入 'exit' 或 'q' 退出)")
    while True:
        try:
            user_input = input("\nEnter your prompt > ")
            if user_input.strip().lower() in ["exit", "q"]:
                break
            if not user_input.strip():
                continue
            generate_and_diagnose(model, tokenizer, user_input)
        except (KeyboardInterrupt, EOFError):
            break
    print("\n测试会话结束！")


if __name__ == "__main__":
    main()
