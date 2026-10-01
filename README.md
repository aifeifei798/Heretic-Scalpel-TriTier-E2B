# 🗡️ Heretic-Scalpel-TriTier-E2B: 大中小三层蜂群架构使用指南

本项目实现了一套创新的 **大中小三层蜂群式混合专家架构（Tri-Tier Swarm MoE）**，基于 **Gemma 4 (E2B)** 底座构建。

该架构彻底摒弃了传统笨重的“全参深拷贝”或“单一大 LoRA 地毯式微调”，通过 **“大核定基盘（只读底座） + 中核管大纲（Dense LoRA） + 32 工蜂做微操（Sparse Micro-LoRA）”**，在单张消费级显卡（RTX 4090 / 5090）上实现了文理思维的连续软插值与纳秒级稀疏激活。

---

## 📂 仓库文件清单与职责

| 文件名 | 职责与作用 |
| :--- | :--- |
| **`build_pristine_dataset.py`** | **数据构建**：生成/清洗 50/50 文理对比语料并为 32 个微专家自动映射分类打标 |
| **`train_scalpel_tritier_e2b.py`** | **极速训练**：原生 BF16 + Rank-64 中核 + 32 微专家小核训练流水线 |
| **`test_scalpel_dualbig.py`** | **本地评测**：直接挂载本地底座与 `.pt` 权重的全功能自动化对照评测终端 |
| **`export_to_hf.py`** | **发布封装**：将训练权重拍平转为 **121 MB** 的官方 `SafeTensors` 格式并生成 `auto_map` 规范代码 |
| **`test-chat.py`** | **生产验证**：模拟远端用户使用 `trust_remote_code=True` 调用原生全息仪表盘进行对话 |
| **`LICENSE`** | Apache 2.0 开源许可证 |

---

## 🛠️ 环境准备与安装

推荐在 Linux 环境（Ubuntu 22.04+）与 Python 3.10+ 环境下运行：

```bash
# 1. 克隆本仓库
git clone https://github.com/aifeifei798/Heretic-Scalpel-TriTier-E2B.git
cd Heretic-Scalpel-TriTier-E2B

# 2. 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers safetensors accelerate
```

---

## 🚀 完整全流程运行指南 (Step-by-Step)

### 第一步：准备训练数据

运行数据生成脚本，生成文理对抗对齐语料 `dual_contrast_data.jsonl`：
```bash
python build_pristine_dataset.py
```
* **输出**：在当前目录下生成包含 `big_target`（0: 文科, 1: 理科）与 `little_group`（0~31 微专家）标签的 `dual_contrast_data.jsonl`。

---

### 第二步：启动三层蜂群极速微调

运行训练脚本，启动大中小三层架构的联合特训：
```bash
python train_scalpel_tritier_e2b.py
```

* **显存占用**：约 17.4 GB（在 24G/32G 卡上运行极度平稳）。
* **训练速度**：RTX 5090 D 达到 **~9.1 samples/s**。
* **黄金停机机制（Early Stopping）**：脚本内置了 **Step 50 强制收工**，总耗时仅需 **1.47 分钟**。
  > ⚠️ **注意**：切勿盲目跑满几百步！在 Step 50 左右，LM Loss 降至 ~1.17，路由器分类边界确立，此时模型泛化灵性最强；若硬跑数百步会导致 LM Loss 暴跌至 0.0001，引发灾难性背诵复读。
* **产出**：生成核心权重文件 `scalpel_tritier_e2b_weights.pt`。

---

### 第三步：本地快速对比评测（直接载入 `.pt`）

在导出前，可先用测试脚本检验文理大核与 32 个微专家的协同表现：
```bash
python test_scalpel_dualbig.py
```

该脚本包含两个阶段：
1. **Phase 1**：自动运行三组预设经典基准题（七言绝句、Python 并发优先队列、瑞利散射物理机制）；
2. **Phase 2**：进入交互式对话终端，每轮对话后实时输出**全息能耗雷达战报**：
   ```text
   ════════════════════════════════════════════════════════════════════════
   🌌【E2B 大中小三层蜂群架构 · 全息透视战报】:
      🏛️  大核 (文科只读底座):     43.5% [██████████            ]
      🔬 中核 (理科 Dense LoRA):  56.5% [████████████          ]
   ────────────────────────────────────────────────────────────────────────
   🪐【小核微专家命中热度榜 (Top 5 Active Micro-Experts)】:
      ✨ #07 [文科组 | Arts_Linguistics]:  2,937 次路由调用
      ✨ #02 [文科组 | Arts_Fiction    ]:  2,009 次路由调用
      ✨ #00 [文科组 | Arts_Prose      ]:  1,796 次路由调用
      ✨ #09 [文科组 | Arts_Ethics     ]:  1,187 次路由调用
      ✨ #01 [文科组 | Arts_Poetry     ]:  1,090 次路由调用
   ════════════════════════════════════════════════════════════════════════
   ```

---

### 第四步：打包成 Hugging Face 官方标准发布包

将训练得到的 `.pt` 拍平转为 Hugging Face 标准的 **`model.safetensors`** 格式，并生成支持 `trust_remote_code=True` 的代码包：

```bash
python export_to_hf.py
```

执行后会全自动完成以下操作：
1. 将嵌套的 PyTorch 张量展平，写入 `./Heretic-Scalpel-TriTier-2B/model.safetensors`（**仅约 121.6 MB**）；
2. 注入具备 `reset_stats()` 与 `show_dashboard()` 原生方法的 `modeling_scalpel.py`；
3. 生成带有 `auto_map` 路由的 `config.json` 及分词器配置；
4. **自动执行沙箱自检**，确认远端用户可一键加载。

打包完毕后的目录结构：
```text
./Heretic-Scalpel-TriTier-2B/
├── config.json                     # 核心 auto_map 映射
├── configuration_scalpel.py        # 自定义 Config
├── modeling_scalpel.py             # 原生内置全息仪表盘的模型架构
├── model.safetensors               # 仅 121.6 MB 的特训器官权重
├── tokenizer.json                  # 分词词表
├── tokenizer_config.json           # 对话模板
├── special_tokens_map.json         # 特殊标记
└── README.md                       # 说明文档
```

---

### 第五步：体验原生仪表盘对话验证

运行 `test-chat.py`，体验如正规 Hugging Face 官方模型般的优雅调用：
```bash
python test-chat.py
```

#### 代码调用范例：
```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./Heretic-Scalpel-TriTier-2B"

# 1. 载入模型（原生支持 trust_remote_code）
tokenizer = AutoTokenizer.from_pretrained(
    model_id, fix_mistral_regex=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)

# 2. 统计前重置探针
model.reset_stats()

# 3. 构造问题并生成
prompt = "用 C++20 实现一个无锁环形缓冲区，必须处理伪共享（False Sharing）问题。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

t0 = time.perf_counter()
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.18,
)
elapsed_sec = time.perf_counter() - t0

gen_ids = outputs[0][inputs.input_ids.shape[1] :]
speed = len(gen_ids) / elapsed_sec if elapsed_sec > 0 else 0

print(tokenizer.decode(gen_ids, skip_special_tokens=True))

# 4. 一行代码呼出原生全息仪表盘
model.show_dashboard(
    speed=speed, gen_tokens=len(gen_ids), elapsed_sec=elapsed_sec
)
```

---

## 🎛️ 核心架构超参数调节说明

若需调整模型的风格倾斜度或抑制程度，可在 `./Heretic-Scalpel-TriTier-2B/config.json` 或 `modeling_scalpel.py` 中微调以下两个参数：

* **`residual_scale` (默认 `0.02`)**：小核微专家残差缩放因子。
  * `0.01 ~ 0.03`：黄金平衡点。既有微专家的高精度纠偏，又保证回答如行云流水，杜绝复读。
  * `> 0.1`：强化垂直专家特异性，但在小样本过度拟合时容易增加词汇引力。
* **`sci_scale` (默认 `0.1`)**：理科中核干预系数。
  * 控制 Dense LoRA 对底座的扰动强度，保证理科逻辑接入的同时不冲淡文科语感。

---

## 🌐 一键推送到 Hugging Face Hub

只需使用 Hugging Face 官方 CLI 即可将打包好的轻量仓库秒级推送：

```bash
huggingface-cli login
huggingface-cli upload your-username/Heretic-Scalpel-TriTier-E2B ./Heretic-Scalpel-TriTier-2B .
```

推送完成后，全球开发者即可直接通过：
```python
AutoModelForCausalLM.from_pretrained(
    "your-username/Heretic-Scalpel-TriTier-E2B", trust_remote_code=True
)
```
免配置直接调用这套大中小三层蜂群模型！
