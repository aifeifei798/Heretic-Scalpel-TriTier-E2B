import json
import os
import shutil
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer

EXPORT_DIR = "./Heretic-Scalpel-TriTier-2B"
BASE_MODEL_ID = "aifeifei798/Heretic-Scalpel-E2B"
PT_WEIGHTS_PATH = "scalpel_tritier_e2b_weights.pt"

os.makedirs(EXPORT_DIR, exist_ok=True)
print("=" * 72)
print(f"📦 正在为【Heretic-Scalpel-TriTier-2B】注入原生全息仪表盘引擎...")
print("=" * 72)

# 清理缓存防止旧模块残留
cache_dir = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules")
if os.path.exists(cache_dir):
    for item in os.listdir(cache_dir):
        if "Scalpel" in item or "TriTier" in item:
            shutil.rmtree(os.path.join(cache_dir, item), ignore_errors=True)

# ----------------------------------------------------------------------
# 1. 确保 model.safetensors 存在（若无则转换）
# ----------------------------------------------------------------------
safetensors_path = os.path.join(EXPORT_DIR, "model.safetensors")
if not os.path.exists(safetensors_path):
    print(f"[*] 正在将 {PT_WEIGHTS_PATH} 转为 SafeTensors 格式...")
    pt_data = torch.load(PT_WEIGHTS_PATH, map_location="cpu")
    flat_safetensors_data = {}
    for key, sub_dict in pt_data.items():
        if isinstance(sub_dict, dict):
            for param_name, tensor in sub_dict.items():
                flat_safetensors_data[f"{key}.{param_name}"] = (
                    tensor.contiguous())
        else:
            flat_safetensors_data[key] = sub_dict.contiguous()
    save_file(flat_safetensors_data, safetensors_path)
    print("[✔] model.safetensors 准备就绪！")
else:
    print("[✔] 检测到已存在 model.safetensors，跳过耗时转换！")

# ----------------------------------------------------------------------
# 2. 生成 configuration_scalpel.py
# ----------------------------------------------------------------------
config_py = """from transformers import PretrainedConfig

class ScalpelTriTierConfig(PretrainedConfig):
    model_type = "scalpel_tritier"

    def __init__(
        self,
        base_model_name_or_path="aifeifei798/Heretic-Scalpel-E2B",
        hidden_dim=1536,
        sci_rank=64,
        micro_rank=16,
        num_experts=32,
        residual_scale=0.02,
        sci_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_dim = hidden_dim
        self.sci_rank = sci_rank
        self.micro_rank = micro_rank
        self.num_experts = num_experts
        self.residual_scale = residual_scale
        self.sci_scale = sci_scale
"""

with open(os.path.join(EXPORT_DIR, "configuration_scalpel.py"),
          "w",
          encoding="utf-8") as f:
    f.write(config_py)
print("[✔] configuration_scalpel.py 写入成功！")

# ----------------------------------------------------------------------
# 3. 生成 modeling_scalpel.py (原生内置 reset_stats 和 show_dashboard)
# ----------------------------------------------------------------------
modeling_py = """from collections import Counter
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PreTrainedModel, AutoModelForCausalLM
from .configuration_scalpel import ScalpelTriTierConfig

EXPERT_NAMES = {
    0: "Arts_Prose", 1: "Arts_Poetry", 2: "Arts_Fiction", 3: "Arts_Drama",
    4: "Arts_Essay", 5: "Arts_History", 6: "Arts_Culture", 7: "Arts_Linguistics",
    8: "Arts_Philosophy", 9: "Arts_Ethics", 10: "Arts_Rhetoric", 11: "Arts_Translation",
    12: "Arts_Critique", 13: "Arts_Dialogue", 14: "Arts_Summary", 15: "Arts_Chat",
    16: "Code_Algo", 17: "Code_DS", 18: "Code_Debug", 19: "Code_Arch",
    20: "Code_Syntax", 21: "Code_Optim", 22: "Math_Algebra", 23: "Math_Geo",
    24: "Math_Prob", 25: "Math_Arith", 26: "Math_Calculus", 27: "Math_Logic",
    28: "Sci_Physics", 29: "Sci_Chem", 30: "Sci_Biology", 31: "Sci_General"
}

class TelemetryManager:
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

class ScalpelLoRAModule(nn.Module):
    def __init__(self, hidden_dim, rank=16, lora_alpha=16.0, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class ScalpelTriTierWrapper(nn.Module):
    def __init__(
        self,
        original_mlp,
        hidden_dim,
        telemetry,
        sci_rank=64,
        micro_rank=16,
        num_experts=32,
        residual_scale=0.02,
        sci_scale=0.1,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.telemetry = telemetry
        self.residual_scale = residual_scale
        self.sci_scale = sci_scale
        self.big_arts = original_mlp

        self.sci_dense_lora = ScalpelLoRAModule(
            hidden_dim=hidden_dim,
            rank=sci_rank,
            lora_alpha=float(sci_rank * 2),
            device=device,
            dtype=dtype
        )

        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_little = nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
        self.lora_pool = nn.ModuleList([
            ScalpelLoRAModule(
                hidden_dim=hidden_dim,
                rank=micro_rank,
                lora_alpha=float(micro_rank * 1),
                device=device,
                dtype=dtype
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        arts_out = self.big_arts(x)
        sci_out = arts_out + self.sci_scale * self.sci_dense_lora(x)

        router_big_logits = self.router_big(x)
        weights_big = torch.softmax(router_big_logits, dim=-1)
        big_tier_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] * sci_out)

        router_little_logits = self.router_little(x)
        chosen_experts = torch.argmax(router_little_logits, dim=-1)

        # 遥测记录
        if self.telemetry is not None:
            self.telemetry.record_step(weights_big, chosen_experts)

        top1_expert_id = chosen_experts[0, -1].item()
        micro_out = self.lora_pool[top1_expert_id](x)

        return big_tier_out + self.residual_scale * micro_out

def locate_layers(model):
    for path in [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            return path(model)
        except Exception:
            pass
    raise RuntimeError("无法定位模型解码器层结构！")

class ScalpelTriTierForCausalLM(PreTrainedModel):
    config_class = ScalpelTriTierConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = None
        self.telemetry = TelemetryManager()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = ScalpelTriTierConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.bfloat16))
        device_map = kwargs.pop("device_map", "cuda:0")

        model_instance = cls(config)
        print(f"[*] 挂载底层大核基座: {config.base_model_name_or_path} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        for attr in ["vision_tower", "audio_tower", "visual", "audio", "mm_projector"]:
            for parent in [base_model, getattr(base_model, "model", None)]:
                if parent is not None and hasattr(parent, attr):
                    setattr(parent, attr, None)
        torch.cuda.empty_cache()

        layers = locate_layers(base_model)
        device = next(base_model.parameters()).device

        print(f"[*] 装配大中小三层蜂群架构 ({len(layers)} 层)...")
        for layer in layers:
            layer.mlp = ScalpelTriTierWrapper(
                original_mlp=layer.mlp,
                hidden_dim=config.hidden_dim,
                telemetry=model_instance.telemetry,
                sci_rank=config.sci_rank,
                micro_rank=config.micro_rank,
                num_experts=config.num_experts,
                residual_scale=config.residual_scale,
                sci_scale=config.sci_scale,
                device=device,
                dtype=dtype
            )

        if os.path.isdir(pretrained_model_name_or_path):
            weights_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if not os.path.exists(weights_file):
                weights_file = os.path.join(pretrained_model_name_or_path, "scalpel_tritier_e2b_weights.pt")
        else:
            from huggingface_hub import hf_hub_download
            try:
                weights_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors")
            except Exception:
                weights_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="scalpel_tritier_e2b_weights.pt")

        print(f"[*] 正在从 {weights_file} 极速载入 SafeTensors 器官权重...")
        if weights_file.endswith(".safetensors"):
            raw_weights = load_file(weights_file, device=str(device))
            for i, layer in enumerate(layers):
                sci_dict = {k.replace(f"layer_{i}_sci_dense_lora.", ""): v for k, v in raw_weights.items() if k.startswith(f"layer_{i}_sci_dense_lora.")}
                layer.mlp.sci_dense_lora.load_state_dict(sci_dict)

                r_big_dict = {k.replace(f"layer_{i}_router_big.", ""): v for k, v in raw_weights.items() if k.startswith(f"layer_{i}_router_big.")}
                layer.mlp.router_big.load_state_dict(r_big_dict)

                r_lit_dict = {k.replace(f"layer_{i}_router_little.", ""): v for k, v in raw_weights.items() if k.startswith(f"layer_{i}_router_little.")}
                layer.mlp.router_little.load_state_dict(r_lit_dict)

                micro_dict = {k.replace(f"layer_{i}_micro_loras.", ""): v for k, v in raw_weights.items() if k.startswith(f"layer_{i}_micro_loras.")}
                layer.mlp.lora_pool.load_state_dict(micro_dict)
        else:
            trained_weights = torch.load(weights_file, map_location=device)
            for i, layer in enumerate(layers):
                layer.mlp.sci_dense_lora.load_state_dict(trained_weights[f"layer_{i}_sci_dense_lora"])
                layer.mlp.router_big.load_state_dict(trained_weights[f"layer_{i}_router_big"])
                layer.mlp.router_little.load_state_dict(trained_weights[f"layer_{i}_router_little"])
                layer.mlp.lora_pool.load_state_dict(trained_weights[f"layer_{i}_micro_loras"])

        base_model.eval()
        base_model.config.use_cache = True
        model_instance.model = base_model
        return model_instance

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    # -------------------------------------------------------------
    # 原生全息仪表盘与探针接口
    # -------------------------------------------------------------
    def reset_stats(self):
        \"\"\"重置遥测探针\"\"\"
        self.telemetry.reset()

    def show_dashboard(self, speed=0.0, gen_tokens=0, elapsed_sec=0.0):
        \"\"\"打印科技感全息遥测战报\"\"\"
        tot = (self.telemetry.arts_weights_sum + self.telemetry.sci_weights_sum) or 1.0
        arts_pct = (self.telemetry.arts_weights_sum / tot) * 100.0
        sci_pct = (self.telemetry.sci_weights_sum / tot) * 100.0

        def make_bar(pct, length=22):
            filled = int(round((pct / 100.0) * length))
            return "█" * filled + " " * (length - filled)

        elapsed_ms = elapsed_sec * 1000.0
        print("\\n" + "═" * 72)
        print("🌌【E2B 大中小三层蜂群架构 · 原生全息透视战报】:")
        if speed > 0 or gen_tokens > 0:
            print(f"   ⚡ 运行状态: {speed:.1f} tokens/s ({gen_tokens} tokens, 耗时 {elapsed_ms:.0f} ms)")
        print(f"   🏛️  大核 (文科只读底座):     {arts_pct:5.1f}% [{make_bar(arts_pct)}]")
        print(f"   🔬 中核 (理科 Dense LoRA):  {sci_pct:5.1f}% [{make_bar(sci_pct)}]")
        print("─" * 72)
        print("🪐【小核微专家命中热度榜 (Top 5 Active Micro-Experts)】:")

        top_active = self.telemetry.expert_activation_counter.most_common(5)
        if top_active:
            for exp_id, count in top_active:
                name = EXPERT_NAMES.get(exp_id, f"Expert_{exp_id}")
                group_tag = "文科组" if exp_id < 16 else "理科组"
                print(f"   ✨ #{exp_id:02d} [{group_tag} | {name:<16}]: {count:>6,d} 次路由调用")
        else:
            print("   (未捕获微专家调用)")
        print("═" * 72 + "\\n")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
"""

with open(os.path.join(EXPORT_DIR, "modeling_scalpel.py"),
          "w",
          encoding="utf-8") as f:
    f.write(modeling_py)
print("[✔] modeling_scalpel.py 写入成功！(已集成 reset_stats 与 show_dashboard)")

# ----------------------------------------------------------------------
# 4. 生成 config.json 与 Tokenizer
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(EXPORT_DIR)

config_dict = {
    "architectures": ["ScalpelTriTierForCausalLM"],
    "model_type": "scalpel_tritier",
    "base_model_name_or_path": BASE_MODEL_ID,
    "hidden_dim": 1536,
    "sci_rank": 64,
    "micro_rank": 16,
    "num_experts": 32,
    "residual_scale": 0.02,
    "sci_scale": 0.1,
    "auto_map": {
        "AutoConfig": "configuration_scalpel.ScalpelTriTierConfig",
        "AutoModelForCausalLM": "modeling_scalpel.ScalpelTriTierForCausalLM",
    },
}
with open(os.path.join(EXPORT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_dict, f, indent=2, ensure_ascii=False)
print("[✔] config.json 更新完成！")
print("=" * 72)
print(f"🎉 原生遥测仪表盘注入完毕！可以直接运行测试脚本了！")
print("=" * 72)