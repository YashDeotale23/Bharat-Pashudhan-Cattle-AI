import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. LOAD YOUR MODEL
model = load_model('bharat_pashudhan_final.keras', compile=False)

# 2. BREED DATABASE (40 Indigenous Breeds)
BREED_INFO = {
    "ayrshire": {"Origin": "Scotland", "Type": "Dairy", "Milk Yield": "5000kg", "Unique Feature": "Red/white patches."},
    "bargur": {"Origin": "Tamil Nadu", "Type": "Draught", "Milk Yield": "350kg", "Unique Feature": "Brown with white patches."},
    "dangi": {"Origin": "Maharashtra", "Type": "Draught", "Milk Yield": "600kg", "Unique Feature": "Rain hardy; spotted coat."},
    "deoni": {"Origin": "Maharashtra", "Type": "Dual", "Milk Yield": "1100kg", "Unique Feature": "White with black spots."},
    "gir": {"Origin": "Gujarat", "Type": "Milch", "Milk Yield": "2000kg+", "Unique Feature": "Convex forehead; half-moon horns."},
    "hallikar": {"Origin": "Karnataka", "Type": "Draught", "Milk Yield": "500kg", "Unique Feature": "Long, vertical horns."},
    "hariana": {"Origin": "Haryana", "Type": "Dual", "Milk Yield": "1500kg", "Unique Feature": "White/grey coat."},
    "himachali pahari": {"Origin": "HP", "Type": "Draught", "Milk Yield": "500kg", "Unique Feature": "Small; cold hardy."},
    "kangayam": {"Origin": "Tamil Nadu", "Type": "Draught", "Milk Yield": "600kg", "Unique Feature": "Strong build; Jallikattu breed."},
    "kankrej": {"Origin": "Gujarat", "Type": "Dual", "Milk Yield": "1700kg", "Unique Feature": "Large lyre-shaped horns."},
    "khariar": {"Origin": "Odisha", "Type": "Draught", "Milk Yield": "450kg", "Unique Feature": "Small; brown coat."},
    "khillari": {"Origin": "Maharashtra", "Type": "Draught", "Milk Yield": "450kg", "Unique Feature": "Long, pointed horns."},
    "konkan kapila": {"Origin": "Maharashtra", "Type": "Dual", "Milk Yield": "450kg", "Unique Feature": "Reddish-brown/black."},
    "kosali": {"Origin": "Chhattisgarh", "Type": "Draught", "Milk Yield": "200kg", "Unique Feature": "Small; disease resistant."},
    "krishna_valley": {"Origin": "Karnataka", "Type": "Dual", "Milk Yield": "900kg", "Unique Feature": "Massive river-plowing frame."},
    "ladakhi": {"Origin": "Ladakh", "Type": "Dual", "Milk Yield": "300kg", "Unique Feature": "Thick hair for cold."},
    "lakhimi": {"Origin": "Assam", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Hardy North-East breed."},
    "malnad_gidda": {"Origin": "Karnataka", "Type": "Milch", "Milk Yield": "300kg", "Unique Feature": "Medicinal milk dwarf."},
    "mewati": {"Origin": "Rajasthan", "Type": "Dual", "Milk Yield": "1000kg", "Unique Feature": "Quiet temperament."},
    "nari": {"Origin": "Rajasthan", "Type": "Dual", "Milk Yield": "1100kg", "Unique Feature": "White/grey; desert hardy."},
    "ongole": {"Origin": "Andhra Pradesh", "Type": "Dual", "Milk Yield": "1500kg", "Unique Feature": "Massive hump; heat tolerant."},
    "poda thirupu": {"Origin": "Telangana", "Type": "Draught", "Milk Yield": "500kg", "Unique Feature": "Forest transport specialist."},
    "punganur": {"Origin": "Andhra Pradesh", "Type": "Milch", "Milk Yield": "500kg", "Unique Feature": "Shortest cattle breed."},
    "purnea": {"Origin": "Bihar", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Adapted to humid plains."},
    "red_sindhi": {"Origin": "Sindh/India", "Type": "Milch", "Milk Yield": "2000kg", "Unique Feature": "Deep dark red color."},
    "sahiwal": {"Origin": "Punjab", "Type": "Milch", "Milk Yield": "2500kg+", "Unique Feature": "Loose skin; high yield."},
    "tharparkar": {"Origin": "Rajasthan", "Type": "Dual", "Milk Yield": "1800kg", "Unique Feature": "Drought hardy; white coat."},
    "umblachery": {"Origin": "Tamil Nadu", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Calves born red, turn grey."},
    "bhelai": {"Origin": "Chhattisgarh", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Light grey; medium size."},
    "dagri": {"Origin": "Gujarat", "Type": "Draught", "Milk Yield": "300kg", "Unique Feature": "Hilly tribal area specialist."},
    "gangatari": {"Origin": "UP/Bihar", "Type": "Dual", "Milk Yield": "1000kg", "Unique Feature": "Found on Ganges banks."},
    "gaolao": {"Origin": "Maharashtra", "Type": "Dual", "Milk Yield": "800kg", "Unique Feature": "Fast walking speed."},
    "ghumsari": {"Origin": "Odisha", "Type": "Draught", "Milk Yield": "500kg", "Unique Feature": "Paddy field worker."},
    "kherigarh": {"Origin": "UP", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Very active; white coat."},
    "malvi": {"Origin": "MP", "Type": "Draught", "Milk Yield": "800kg", "Unique Feature": "Deep grey; strong legs."},
    "motu": {"Origin": "Odisha", "Type": "Draught", "Milk Yield": "200kg", "Unique Feature": "Tiny; low maintenance."},
    "nagori": {"Origin": "Rajasthan", "Type": "Draught", "Milk Yield": "500kg", "Unique Feature": "Best speed/trotting breed."},
    "ponwar": {"Origin": "UP", "Type": "Draught", "Milk Yield": "400kg", "Unique Feature": "Black and white patches."},
    "siri": {"Origin": "Sikkim", "Type": "Dual", "Milk Yield": "800kg", "Unique Feature": "Himalayan giant; thick hair."},
    "thutho": {"Origin": "Nagaland", "Type": "Draught", "Milk Yield": "300kg", "Unique Feature": "Rainforest hardy breed."}
}

CLASSES = ['Ayrshire', 'Bargur', 'Dangi', 'Deoni', 'Gir', 'Hallikar', 'Hariana', 'Himachali Pahari', 'Kangayam', 'Kankrej', 'Khariar', 'Khillari', 'Konkan Kapila', 'Kosali', 'Krishna_Valley', 'Ladakhi', 'Lakhimi', 'Malnad_gidda', 'Mewati', 'Nari', 'Ongole', 'Poda Thirupu', 'Punganur', 'Purnea', 'Red_Sindhi', 'Sahiwal', 'Tharparkar', 'Umblachery', 'bhelai', 'dagri', 'gangatari', 'gaolao', 'ghumsari', 'kherigarh', 'malvi', 'motu', 'nagori', 'ponwar', 'siri', 'thutho'] 

# 3. PREDICTION LOGIC
def predict_cattle(img):
    if img is None: 
        return "## ⚠️ Please upload an image.", None

    # Version A: Original (299x299)
    v1 = tf.image.resize(img, (299, 299))
    
    # Version B: Horizontal Flip
    v2 = tf.image.flip_left_right(v1)
    
    # Version C: Central Zoom (0.80)
    v3 = tf.image.central_crop(v1, central_fraction=0.8)
    v3 = tf.image.resize(v3, (299, 299))

    # Inference (1/255 scale)
    batch = np.array([v1, v2, v3]) / 255.0
    preds_all = model.predict(batch, verbose=0)
    avg_preds = np.mean(preds_all, axis=0)
    
    top_idx = np.argmax(avg_preds)
    conf = float(avg_preds[top_idx])

    # --- THE 25% BARRIER ---
    if conf < 0.25:
        msg = "## 🔍 Breed Not Recognized\n"
        msg += f"The AI is only {conf*100:.1f}% confident, which is below our safety threshold.\n\n"
        msg += "**Possible reasons:**\n* The image is not a recognized indigenous cattle breed.\n* The photo is too blurry or the animal is too far away."
        return msg, {"Unrecognized": conf}

    # Format Success Result
    raw_name = CLASSES[top_idx]
    lookup_name = raw_name.strip().lower()

    info = BREED_INFO.get(lookup_name, {"Origin": "Unknown", "Type": "Unknown", "Milk Yield": "N/A", "Unique Feature": "Native Breed"})
    
    result_text = f"## 🏆 Result: {raw_name.upper()}\n"
    result_text += f"**AI Confidence:** {conf*100:.2f}%\n"
    result_text += f"*(Processed via TTA-Ensemble for High Accuracy)*\n\n"
    result_text += f"📍 **Origin:** {info['Origin']} | 🥛 **Category:** {info['Type']}\n"
    result_text += f"📊 **Yield:** {info['Milk Yield']} | ⭐ **Feature:** {info['Unique Feature']}"

    # Top 3 scores for the chart
    top_3_indices = np.argsort(avg_preds)[-3:][::-1]
    top_3_confidences = {CLASSES[i]: float(avg_preds[i]) for i in top_3_indices}

    return result_text, top_3_confidences

# 4. UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌾 Bharat Pashudhan: Smart Breed Identification")
    gr.Markdown("Identify 40 different Indian cattle breeds using advanced Computer Vision.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Cow Image", type="numpy")
            run_btn = gr.Button("🔍 IDENTIFY BREED", variant="primary")
        
        with gr.Column():
            output_md = gr.Markdown(value="Results will appear here...")
            output_chart = gr.Label(label="Confidence Distribution", num_top_classes=3)

    run_btn.click(fn=predict_cattle, inputs=[input_img], outputs=[output_md, output_chart])

# Line 119
if __name__ == "__main__":
    # Line 121 (This line MUST be indented/pushed to the right)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)