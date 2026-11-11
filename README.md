End-to-End LLM Post-Training Pipeline: SFT + DPO for Deterministic ExtractionThis repository contains the end-to-end implementation of an LLM post-training alignment pipeline that adapts Qwen-2.5-7B-Instruct into a deterministic, zero-hallucination information extraction engine.Using QLoRA (NF4 quantization) with Unsloth and Hugging Face TRL, the pipeline trains only 0.53% of the total model weights on a single commodity GPU (NVIDIA Tesla T4, 15 GB VRAM), advancing through Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and automated Pydantic validation benchmarks.📌 Executive Summary & Key ResultsMetric / DimensionBaseline (Zero-Shot Qwen-2.5)Post-SFT CheckpointPost-DPO Final CheckpointStrict Schema Compliance0.0% (Markdown tags, commentary)60.0% (Repetition, key drift)100.0% (5/5 holdout pass)Pydantic Parse Error RateHigh (KeyErrors on nested structures)Moderate (Duplicate code blocks)0.0%Training Parameter BudgetN/A40,370,176 / 7,655,986,6880.53% trainedSFT Loss ConvergenceN/AStep 1: 4.187 $\to$ Step 30: 0.018Loss: 0.0189DPO Reward MarginN/AStep 3: 0.066Step 15: 2.834 (100% accuracy)Inference Footprint~14.5 GB (FP16)~5.8 GB (4-bit NF4 quantized)~5.8 GB VRAM🏗 Architecture & Engineering PipelineRaw Unstructured Invoice / Note
       │
       ▼
┌────────────────────────────────────────┐
│ 1. Pydantic Rejection Sampling         │ ───► Filter malformed samples / enforce schema
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 2. Parameter-Efficient QLoRA SFT       │ ───► Teach ChatML syntax & flat JSON representation
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 3. Direct Preference Optimization (DPO)│ ───► Suppress repetition, preambles & key hallucinations
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 4. Deterministic Greedy Inference      │ ───► Generate with explicit `<|im_end|>` termination
└────────────────────────────────────────┘
       │
       ▼
Interactive Gradio Web Application + Pydantic Verification
1. Data Integrity via PydanticUnstructured extraction pipelines in production break when LLMs arbitrarily change dictionary keys or nest fields inside artificial wrappers ({"invoice": {"vendor": ...}}). We enforce a strict runtime type contract using Pydantic:from pydantic import BaseModel

class InvoiceExtraction(BaseModel):
    vendor_name: str
    invoice_date: str
    total_amount_usd: float
    items_count: int
During synthetic data generation, any generation that fails parsing or schema validation is discarded via rejection sampling.2. QLoRA Mathematical FormulationTo fine-tune a 7.6-billion-parameter model on a single free-tier 15 GB GPU, the base model is quantized to 4-bit NormalFloat (NF4). The original weight matrix $W_0 \in \mathbb{R}^{d \times k}$ remains completely frozen. Trainable rank-decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are injected:$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$Where:Rank $r = 16$LoRA Alpha scaling constant $\alpha = 32$LoRA Dropout $= 0$ (optimized for Unsloth kernel fusion)Target modules: Attention projections (q_proj, k_proj, v_proj, o_proj) and MLP layers (gate_proj, up_proj, down_proj).3. Contrastive Direct Preference Optimization (DPO)While SFT teaches the model the target domain vocabulary, it frequently outputs conversational preambles ("Sure! Here is your JSON:"), markdown ticks (json ... ), and duplicate generations.Instead of training an unstable Reinforcement Learning from Human Feedback (RLHF) reward model, we optimize using Direct Preference Optimization (DPO) directly on preference triplets:Prompt ($x$): Invoice text formatted with ChatML syntax.Chosen ($y_w$): Minimalist, flat JSON string with explicit <|im_end|> termination.Rejected ($y_l$): Markdown-wrapped code blocks, conversational greetings, and nested hallucinations.The DPO loss directly increases the relative log probability of $y_w$ over $y_l$:$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w\vert{}x)}{\pi_{\text{ref}}(y_w\vert{}x)} - \beta \log \frac{\pi_\theta(y_l\vert{}x)}{\pi_{\text{ref}}(y_l\vert{}x)} \right) \right]$$We set $\beta = 0.1$ and a reduced learning rate of $5 \times 10^{-6}$ to prevent catastrophic forgetting.📈 Training Telemetry & LogsSFT Convergence (Epoch 15/15)Step    Training Loss
5       2.563272
10      0.931441
15      0.270593
20      0.084618
25      0.027270
30      0.018952 (Smooth convergence)
DPO Preference Dynamics (Epoch 15/15)Step   Loss       Reward Chosen   Reward Rejected   Reward Margin   Accuracy
3      0.6615     +0.0018         -0.0651           +0.0669         33.3%
6      0.3474     +0.0315         -0.9176           +0.9491         100.0%
9      0.1724     +0.0650         -1.7899           +1.8549         100.0%
12     0.1048     +0.0919         -2.3901           +2.4821         100.0%
15     0.0783     +0.1067         -2.7278           +2.8346         100.0%
🧪 Evaluation on Unseen Holdout Test SuiteWe evaluated the final model on 5 structurally distinct, unseen test instances using deterministic greedy decoding (do_sample=False, temperature=0.0 equivalent):# Unseen Messy Test Input:
"Hey accounting team - forwarding this over!! Just paid CyberShield Networks for the 7 firewall licenses we ordered back on 2025-11-04. They charged the corporate card a grand total of $2840.75 (includes taxes). Please log this ASAP, thanks! - Dave"

# Output from DPO-Aligned Model:
{"vendor_name": "CyberShield Networks", "invoice_date": "2025-11-04", "total_amount_usd": 2840.75, "items_count": 7}

# Pydantic Validation: PASSED (100% schema match)
Holdout Summary:Test 1 (Stellar Cloud Services): PASSEDTest 2 (Quantum Security LLC): PASSEDTest 3 (Horizon Data Corp pipe-separated table): PASSEDTest 4 (Vertex AI Labs): PASSEDTest 5 (Pinnacle Freight): PASSEDOverall Compliance Rate: 5/5 (100.0%)💻 Quickstart & Reproduction1. Clone Repository & Install Dependenciesgit clone https://github.com/YOUR_USERNAME/llm-post-training-sft-dpo.git
cd llm-post-training-sft-dpo

pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt
2. Run the Full Training NotebookLaunch Google Colab or your Jupyter server with a CUDA GPU:jupyter notebook notebooks/post_training_pipeline.ipynb
3. Launch the Interactive Web UIRun the Gradio application to test noisy extraction queries live:python app.py
📂 Repository File Structurellm-post-training-sft-dpo/
├── README.md                              <- Technical documentation & benchmark report
├── requirements.txt                       <- Environment dependencies
├── app.py                                 <- Interactive Gradio deployment application
├── LICENSE                                <- MIT License
└── post_training_pipeline.ipynb       <- Executable Colab notebook with training logs
    
📜 LicenseDistributed under the MIT License. See LICENSE for more information.
