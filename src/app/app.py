from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "optimized_model.pt"

st.set_page_config(page_title="Semne rutiere", page_icon="🚦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=DM+Sans:wght@400;500;700&display=swap');
.stApp { background: linear-gradient(135deg, #f5f1e8 0%, #e5edf0 55%, #d9e6df 100%); color: #17252d; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0; }
.hero { border-left: 8px solid #db3a34; padding: 0.2rem 1.2rem; margin: 1rem 0 2rem; }
.hero p { font-family: 'DM Sans', sans-serif; color: #53656b; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Semne rutiere, vazute clar.</h1><p>Detector YOLO v11 pentru cele 15 clase selectate din GTSRB.</p></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return YOLO(path)


with st.sidebar:
    st.subheader("Detector")
    weights_path = st.text_input("Model .pt", str(DEFAULT_WEIGHTS))
    confidence = st.slider("Prag incredere", 0.05, 0.95, 0.35, 0.05)
    st.info(f"Model optimizat încarcat: {Path(weights_path).name}")
    st.markdown("---")
    st.markdown("**Status Sistem (State Machine):**")
    st.success("STARE: RN_INFERENCE")

model = get_model(weights_path) if Path(weights_path).exists() else None
if model is None:
    st.warning("Modelul antrenat nu exista. Ruleaza modulul de antrenare inainte de inferenta.")

upload = st.file_uploader("Incarca o imagine", type=["jpg", "jpeg", "png", "ppm"])
if upload:
    image = Image.open(upload).convert("RGB")
    if model is None:
        st.image(image, use_container_width=True)
    else:
        # Verificăm încrederea (adăugat în Etapa 6)
        high_conf_threshold = 0.6
        result = model.predict(image, conf=confidence, device="cpu", verbose=False)[0]
        
        left, right = st.columns([1.7, 1])
        with left:
            st.image(result.plot(), channels="BGR", use_container_width=True)
        with right:
            st.subheader("Ce am găsit:")
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf >= high_conf_threshold:
                        st.metric(result.names[class_id], f"{conf:.0%}", delta="Siguranță mare")
                    else:
                        st.metric(result.names[class_id], f"{conf:.0%}", delta="- Siguranță mică", delta_color="inverse")
                        st.warning(f"Nu sunt foarte sigur pe {result.names[class_id]}. Aruncă o privire.")
            else:
                st.caption("Nu a fost gasit niciun semn peste pragul ales.")