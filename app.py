import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# 1. LOAD MODEL
model_id = "Vanu282003/cancer-ai-model"
model = ViTForImageClassification.from_pretrained(model_id, output_attentions=True, low_cpu_mem_usage=True)
processor = ViTImageProcessor.from_pretrained(model_id)

# 2. HEATMAP LOGIC
def generate_heatmap(image, attentions):
    att_mat = torch.stack(attentions).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    res = torch.eye(att_mat.size(-1))
    for a in att_mat:
        res = torch.matmul(a, res)
    
    mask = res[0, 1:].reshape(14, 14).detach().numpy()
    mask = cv2.resize(mask / (mask.max() + 1e-8), (image.size[0], image.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(np.array(image.convert("RGB")), 0.5, heatmap, 0.5, 0)
    return overlay

# 3. PREDICT FUNCTION
def predict(image):
    if image is None: return None, None
    image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        attentions = outputs.attentions
    heatmap_img = generate_heatmap(image, attentions)
    results = {"Healthy": float(probs[0]), "Malignant": float(probs[1])}
    return heatmap_img, results

# 4. AQUA-BLACK UI
aqua_css = """
body, .gradio-container { background-color: #000000 !important; }
.md, p, h1, h2, h3, span, label { color: #00FFFF !important; }
.gr-button-primary { background: #00FFFF !important; color: black !important; }
"""

with gr.Blocks(css=aqua_css) as demo:
    # --- HEADER & DESCRIPTION ---
    gr.Markdown("# 🔬 MEDICAL AI: CANCER DETECTION SYSTEM")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Tissue Patch")
            btn = gr.Button("🚀 START ANALYSIS", variant="primary")
        with gr.Column():
            heatmap_out = gr.Image(label="Explainability Heatmap")
            label_out = gr.Label(num_top_classes=2, label="Prediction")

    # --- HEATMAP GUIDE SECTION ---
    gr.Markdown("---")
    gr.Markdown("## 🔍 How to Read the AI Heatmap")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 🌡️ Color Meaning
            * 🔴 **RED:** High Importance. The AI is focusing on these high-risk areas.
            * 🟡 **YELLOW:** Moderate interest.
            * 🔵 **BLUE:** Ignored/Background tissue.
            """)
        with gr.Column():
            gr.Markdown("""
            ### 🩺 Why this helps?
            Pathologists can verify if the AI is looking at the **actual nuclei** of the cells or just background noise, making the AI more trustworthy.
            """)

    btn.click(fn=predict, inputs=img_input, outputs=[heatmap_out, label_out], queue=False)

demo.launch()