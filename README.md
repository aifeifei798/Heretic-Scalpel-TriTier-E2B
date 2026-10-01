# 🗡️ Heretic-Scalpel-TriTier-E2B: User Manual & Execution Guide

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Backbone-Heretic--Scalpel--E2B-orange.svg)](https://huggingface.co/aifeifei798/Heretic-Scalpel-E2B)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SafeTensors-green.svg)](https://huggingface.co/aifeifei798/Heretic-Scalpel-TriTier-E2B)
[![Architecture](https://img.shields.io/badge/Architecture-Tri--Tier%20Swarm%20MoE-purple.svg)](#architecture-overview)
[![Format](https://img.shields.io/badge/Weights-SafeTensors%20(121.6%20MB)-green.svg)](#step-4-export-to-hugging-face-safetensors-format)

[简体中文](https://github.com/aifeifei798/Heretic-Scalpel-TriTier-E2B/blob/main/README_zh_cn.md)

**Heretic-Scalpel-TriTier-E2B** introduces an innovative **Tri-Tier Swarm Mixture-of-Adapters (Tri-Tier Swarm MoE)** architecture built on top of the **Gemma 4 (E2B)** backbone.

By abandoning traditional full-parameter duplication and monolithic single-LoRA fine-tuning, this project implements a hierarchical paradigm:
1. **Tier-1: Arts Anchor Core (Immutable Read-Only Backbone)** — Preserves foundational world knowledge and linguistic fluency with zero parameter drift.
2. **Tier-2: STEM Mid-Core (Rank-64 Dense LoRA)** — Governs macro-level logical induction, coding syntax, and physical reasoning.
3. **Tier-3: Micro-Expert Swarm (32 Sparsely Activated Rank-16 LoRAs)** — Executes high-precision, sub-space directional steering for specific domains (96 KB per expert).

This repository contains the complete pipeline: from dataset construction and rapid alignment to weight export and real-time holographic telemetry inference.

---

## 📂 Repository File Structure & Roles

| File | Description & Purpose |
| :--- | :--- |
| **`build_pristine_dataset.py`** | **Data Preparation**: Generates and formats the 50/50 Arts vs. STEM contrastive corpus with 32-class micro-expert labels. |
| **`train_scalpel_tritier_e2b.py`** | **Fast Training Pipeline**: High-throughput training using native BF16, Rank-64 STEM Core, and 32 micro-experts. |
| **`test_scalpel_dualbig.py`** | **Scripted Benchmark & Chat**: Local test script evaluating base weights + `.pt` checkpoints with real-time telemetry. |
| **`export_to_hf.py`** | **Packaging & Conversion**: Flattens trained PyTorch weights into a **121.6 MB** Hugging Face `model.safetensors` release with `auto_map`. |
| **`test-chat.py`** | **Production Verification**: Demonstrates plug-and-play loading via `trust_remote_code=True` with native dashboard APIs. |
| **`LICENSE`** | Apache 2.0 Open-Source License. |

---

## 🛠️ Prerequisites & Installation

Recommended environment: **Linux (Ubuntu 22.04+)**, **Python 3.10+**, and an NVIDIA GPU with at least 16 GB VRAM (RTX 4090 / RTX 5090 / A100 / H100 recommended).

```bash
# 1. Clone this repository
git clone https://github.com/aifeifei798/Heretic-Scalpel-TriTier-E2B.git
cd Heretic-Scalpel-TriTier-E2B

# 2. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install core dependencies
pip install transformers safetensors accelerate bitsandbytes
```

---

## 🚀 End-to-End Execution Workflow

### Step 1: Prepare the Contrastive Dataset

Generate the dual-domain contrastive training data (`dual_contrast_data.jsonl`):

```bash
python build_pristine_dataset.py
```

* **Output**: Produces `dual_contrast_data.jsonl` in the current directory.
* Each entry contains:
  * `big_target`: Binary indicator (`0`: Arts & Humanities, `1`: STEM & Logic).
  * `little_group`: Targeted micro-expert routing index (`0` to `31`).

---

### Step 2: Launch Tri-Tier Swarm Training

Train the dual routers, Dense STEM Core, and 32 micro-experts simultaneously:

```bash
python train_scalpel_tritier_e2b.py
```

#### Training Specs & Runtime Profile:
* **VRAM Consumption**: Peak at **~17.4 GB** (runs comfortably on 24 GB / 32 GB GPUs).
* **Throughput**: Reaches **~9.1 samples/s** on an NVIDIA RTX 5090 D.
* **The "Golden Stopping" Zone**:
  The script enforces early stopping at **Step 50** (~1.47 minutes total runtime).
  > ⚠️ **Important Note**: Do **NOT** train this architecture for hundreds of steps on small datasets! Around Step 50, the Language Modeling loss (LM Loss) stabilizes at `~1.17`, and router separation is firmly established. Pushing training to 300+ steps drives LM Loss down to `< 0.0002`, triggering **severe memorization collapse** (verbatim repetition of training samples).
* **Artifact**: Generates `scalpel_tritier_e2b_weights.pt` (~130 MB).

---

### Step 3: Scripted Local Benchmark & Interactive Inspection

Evaluate the trained checkpoint and inspect real-time router decisions:

```bash
python test_scalpel_dualbig.py
```

1. **Phase 1 (Automated Benchmark)**: Runs three diverse prompts:
   - Classical Chinese 7-Character Regulated Verse (*七言绝句*).
   - Python generic concurrent priority queue with Lock-Free/CAS analysis.
   - Rayleigh scattering and atmospheric optical physics.
2. **Phase 2 (Interactive Shell)**: Chat dynamically while observing the holographic telemetry report after every turn:

```text
════════════════════════════════════════════════════════════════════════
🌌【E2B Tri-Tier Swarm Architecture · Holographic Telemetry Report】:
   🏛️  Tier-1 Arts Anchor Core:     43.5% [██████████            ]
   🔬 Tier-2 STEM Dense LoRA:       56.5% [████████████          ]
────────────────────────────────────────────────────────────────────────
🪐【Active Micro-Expert Swarm Radar (Top 5 Active)】:
   ✨ #07 [Arts | Arts_Linguistics]:  2,937 calls
   ✨ #02 [Arts | Arts_Fiction    ]:  2,009 calls
   ✨ #00 [Arts | Arts_Prose      ]:  1,796 calls
   ✨ #09 [Arts | Arts_Ethics     ]:  1,187 calls
   ✨ #01 [Arts | Arts_Poetry     ]:  1,090 calls
════════════════════════════════════════════════════════════════════════
```

---

### Step 4: Export to Hugging Face SafeTensors Format

Convert the checkpoint into a modular, production-ready Hugging Face repository package:

```bash
python export_to_hf.py
```

This automated step:
1. Flattens nested PyTorch tensors and serializes them into **`model.safetensors` (only ~121.6 MB)**.
2. Embeds the custom architecture and native telemetry methods (`reset_stats()`, `show_dashboard()`) into `modeling_scalpel.py`.
3. Injects the `auto_map` configuration into `config.json`.
4. Runs an immediate **in-memory sandbox self-test** to verify compatibility.

#### Generated Directory (`./Heretic-Scalpel-TriTier-2B`):
```text
Heretic-Scalpel-TriTier-2B/
├── config.json                     # AutoMap & model specifications
├── configuration_scalpel.py        # ScalpelTriTierConfig class definition
├── modeling_scalpel.py             # Model code + native telemetry APIs
├── model.safetensors               # 121.6 MB SafeTensors weight file
├── tokenizer.json                  # Gemma-4 fast tokenizer vocabulary
├── tokenizer_config.json           # Chat templates & generation markers
├── special_tokens_map.json         # Special token markers
└── README.md                       # Model documentation card
```

---

### Step 5: Test via Hugging Face `trust_remote_code=True`

Run `test-chat.py` to verify remote-style inference:

```bash
python test-chat.py
```

#### Code Snippet:
```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./Heretic-Scalpel-TriTier-2B"

# 1. Load model and tokenizer natively
tokenizer = AutoTokenizer.from_pretrained(
    model_id, fix_mistral_regex=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True
)

# 2. Reset telemetry probe before generation
model.reset_stats()

# 3. Prompt and run generation
prompt = "Implement a Lock-Free Ring Buffer in C++20 using acquire/release memory order and cache line padding."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

t0 = time.perf_counter()
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.18
)
elapsed_sec = time.perf_counter() - t0

gen_ids = outputs[0][inputs.input_ids.shape[1]:]
speed = len(gen_ids) / elapsed_sec if elapsed_sec > 0 else 0

print(tokenizer.decode(gen_ids, skip_special_tokens=True))

# 4. Display the built-in holographic radar dashboard
model.show_dashboard(speed=speed, gen_tokens=len(gen_ids), elapsed_sec=elapsed_sec)
```

---

## 🎛️ Hyperparameter Tuning Guide

You can tune inference sensitivity inside `./Heretic-Scalpel-TriTier-2B/config.json` or `modeling_scalpel.py`:

* **`residual_scale` (Default: `0.02`)**:
  * Scaling factor for the active micro-expert residual injection.
  * `0.01 ~ 0.03`: **Recommended sweet spot**. Provides subtle, surgical corrections while preserving fluent base model behavior.
  * `> 0.1`: Amplifies micro-expert influence; however, if trained on a small dataset, higher values may pull generation toward memorized patterns.
* **`sci_scale` (Default: `0.1`)**:
  * Scaling factor for the Rank-64 Dense STEM Core.
  * Balances deep mathematical/code rigor with general conversational warmth.

---

## 🌐 Publishing to the Hugging Face Hub

Push your export directly to the Hugging Face Model Hub using the CLI:

```bash
# 1. Log in with your Hugging Face Write Token
huggingface-cli login

# 2. Upload the export directory
huggingface-cli upload your-username/Heretic-Scalpel-TriTier-E2B ./Heretic-Scalpel-TriTier-2B .
```

Once uploaded, anyone can load your model directly with two lines:

```python
model = AutoModelForCausalLM.from_pretrained(
    "your-username/Heretic-Scalpel-TriTier-E2B", trust_remote_code=True
)
```

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## ⚠️ Exhaustive Liability Disclaimer & Safety Notice

### 【 CRITICAL WARNING 】
**Heretic-Scalpel-TriTier-E2B is an advanced, specialized mathematical, causal, and engineering reasoning instrument. It has been completely stripped of corporate content filters, moralizing alignment lectures, and conversational hedging. By downloading, loading, quantizing, integrating, or running inference on these weights, you unconditionally and irrevocably accept the following binding terms. If you do not agree, delete and destroy all copies of this model immediately.**

### 1. Absolute Proscription of CBRN, Munitions & Warfare Systems
You are strictly prohibited from utilizing this model for the research, design, synthesis, optimization, or reverse engineering of conventional, non-conventional, or asymmetric weapons systems, including but not limited to:
* **Nuclear & Radiological Weapons**: Enrichment cascade configurations, fissile critical mass geometries, neutron reflector arrays, or radiological dispersal device (dirty bomb) fluid dispersion modeling.
* **Chemical Warfare Agents & Neurotoxins**: Synthetic pathways, precursor sourcing, or stabilization dynamics for organophosphate nerve agents (such as Sarin, VX, or Novichok-class agents), vesicants (such as sulfur mustard), or toxic industrial chemical weaponization.
* **Biological Pathogens & Gene Editing**: Aerosolized delivery mechanics, virulence enhancement, antibiotic resistance engineering, or genetic targeting of pathogenic agents (including Bacillus anthracis, viral hemorrhagic fevers, or modified orthopoxviruses).
* **Improvised Explosive Devices (IEDs) & High Explosives**: Detonator firing trains, energetic formulation stoichiometries, or Explosively Formed Penetrator (EFP) liner geometry optimizations.
* **Autonomous Lethal Targeting**: Integrating this model into kinetic strike platforms, loitering munitions, automated target tracking systems, or swarming arrays without real-time human command authority.

### 2. Hazardous Physical, Chemical & Experimental Catastrophe Disclaimer
All thermodynamic, fluidic, structural, and mechanical simulations produced by this model exist strictly as mathematical approximations within idealized constraints. Under no circumstances should these outputs serve as certified engineering procedures in life-critical environments:
* **Pressure Vessels & Deep-Submergence Vehicles**: Never deploy real-world submarines, hyperbaric chambers, or pressurized autoclaves based solely on this model's yield-strength, buckling-mode, or hydrostatic calculations.
* **Exothermic & Reactive Chemistry**: Never conduct pressurized hydrogenations, runaway polymerizations, or volatile aerosol mixtures based on reaction dynamics inferred by this model outside of certified containment laboratories.
* **High-Voltage & Pulsed-Power Systems**: Never construct custom transformer taps, uninsulated plasma discharge arcs, or high-energy directed laser apparatus relying on dielectric breakdown estimations generated by this model.

### 3. Absolute Prohibition of Exploitative, Explicit, and Malicious Utility
* **Child Exploitation (CSAM/CSAE)**: Zero-tolerance. You may not utilize, adapt, or prompt this model directly or indirectly to generate, parse, or process material involving the exploitation, abuse, or harm of minors.
* **Non-Consensual Imagery & Impersonation**: Generating synthetic parameters for the non-consensual fabrication of sexually explicit content, defamation, or deepfake extortion is strictly prohibited.
* **Offensive Cyber Operations**: You may not deploy this model to generate zero-day exploits, craft automated polymorphic malware, automate ransomware command architectures, or sabotage industrial SCADA networks.

### 4. Exclusion of Professional, Medical, Legal & Fiduciary Warranties
* **No Medical Practice**: Pathological, physiological, or molecular interpretations provided by this model do not constitute clinical diagnoses, surgical protocols, or therapeutic prescriptions. The authors bear zero liability for self-treatment, medical non-compliance, bodily injury, or death.
* **No Attorney-Client Privilege or Legal Counsel**: Contractual risk evaluations and statutory compliance audits represent syntactic and structural analyses only. They do not constitute formal legal opinions and cannot replace licensed legal representation.
* **No Investment or Fiduciary Advice**: Macroeconomic assessments, trade dispute models, and asset-flow analyses generated by this model do not constitute investment advice, equity ratings, or hedging recommendations.

### 5. Deterministic Voice Disclaimer & Total User Indemnification
* **Technical Definition of "Determinism"**: The assertive, non-hedging tone of this model (*"must," "invariable failure," "structural collapse"*) is an artifact of algorithmic probability optimization designed to purge superficial filler text. It does not represent omniscience, infallibility, or verified physical ground truth.
* **Sole Liability Rests with the User**: The operator assumes 100% full, unmitigated personal, civil, and criminal legal liability for any queries submitted, actions taken, decisions rendered, and physical constructions executed pursuant to outputs of this model. The user agrees to fully defend, indemnify, and hold harmless the model creators, fine-tuning developers, and quantization contributors (including @mradermacher and the open-source community) against all claims, judgments, environmental disasters, financial insolvencies, property destructions, regulatory fines, personal injuries, or fatalities arising out of the deployment of this software.

**THIS MODEL IS PROVIDED "AS-IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, UNDER THE TERMS OF THE APACHE 2.0 LICENSE. YOU HOLD THE BLADE; YOU ASSUME COMPLETE RESPONSIBILITY FOR EVERY CUT.**

