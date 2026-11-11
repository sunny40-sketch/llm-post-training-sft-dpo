import json
import gradio as gr
from pydantic import BaseModel, ValidationError
from unsloth import FastLanguageModel

# 1. LOAD THE TRAINED DPO-ALIGNED MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_dpo_aligned_model",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 2. DEFINE PYDANTIC VALIDATION SCHEMA
class InvoiceExtraction(BaseModel):
    vendor_name: str
    invoice_date: str
    total_amount_usd: float
    items_count: int

# 3. EXTRACTION AND VALIDATION HANDLER
def extract_invoice_app(messy_text):
    if not messy_text or not messy_text.strip():
        return {"error": "Please enter some invoice text."}, "❌ Empty Input"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data extraction system. Output strictly a flat JSON "
                "object with keys: vendor_name, invoice_date, total_amount_usd, items_count."
            )
        },
        {"role": "user", "content": messy_text}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Deterministic greedy decoding via do_sample=False
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=True,
        do_sample=False,
        eos_token_id=tokenizer.encode("<|im_end|>")[0],
    )
    
    decoded = tokenizer.batch_decode(outputs)[0]
    raw_json = decoded.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    # Validate with Pydantic
    try:
        parsed_dict = json.loads(raw_json)
        validated_data = InvoiceExtraction(**parsed_dict)
        return validated_data.model_dump(), "✅ PASSED: 100% Schema Compliant (Pydantic Verified)"
    except (json.JSONDecodeError, ValidationError) as e:
        return {"raw_model_output": raw_json, "validation_error": str(e)}, "❌ FAILED: Schema Validation Error"
    except Exception as e:
        return {"system_error": str(e)}, "⚠️ Runtime Error"

# 4. BUILD GRADIO INTERFACE
with gr.Blocks(title="AI Invoice Extractor") as demo:
    gr.Markdown("# 🧾 AI Invoice Extractor (Qwen-2.5-7B + QLoRA + DPO)")
    gr.Markdown(
        "Paste messy invoice text, emails, or receipts below. The fine-tuned and DPO-aligned "
        "model extracts structured JSON and validates it against a strict Pydantic contract."
    )
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="Messy Invoice Input",
                lines=5,
                value="Hey accounting team - forwarding this over!! Just paid CyberShield Networks for the 7 firewall licenses we ordered back on 2025-11-04. They charged the corporate card a grand total of $2840.75 (includes taxes). Please log this ASAP, thanks! - Dave"
            )
            submit_btn = gr.Button("Extract Structured JSON", variant="primary")
            
            gr.Examples(
                examples=[
                    ["Hey accounting team - forwarding this over!! Just paid CyberShield Networks for the 7 firewall licenses we ordered back on 2025-11-04. They charged the corporate card a grand total of $2840.75 (includes taxes). Please log this ASAP, thanks! - Dave"],
                    ["Invoice from Stellar Cloud Services on 2025-02-14 for 12 server instances totaling $3,400.00."],
                    ["Receipt: Horizon Data Corp | Date: 2025-04-19 | Qty: 6 | Balance Due: $1,920.00"]
                ],
                inputs=input_box
            )
            
        with gr.Column():
            status_box = gr.Textbox(label="Pydantic Validation Status", interactive=False)
            json_output = gr.JSON(label="Extracted JSON Output")

    submit_btn.click(
        fn=extract_invoice_app,
        inputs=input_box,
        outputs=[json_output, status_box]
    )

if __name__ == "__main__":
    demo.launch(share=True)