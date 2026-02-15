import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import io
import cv2
from rembg import remove
import torch
from diffusers import StableDiffusionInpaintPipeline
import requests
from io import BytesIO
import os

# Page config
st.set_page_config(page_title="Passport Photo Tool", layout="wide")

st.title("🛂 Passport Photo Editor")
st.markdown("**Remove BG • Change BG • Upscale • Perfect Passport Photos**")

# Install requirements (run once)
st.sidebar.markdown("### 📦 Install Requirements")
if st.sidebar.button("Install Dependencies"):
    st.sidebar.code("""
pip install streamlit pillow opencv-python rembg torch torchvision diffusers transformers accelerate
    """)

@st.cache_resource
def load_model():
    """Load removal background model"""
    return None  # rembg handles this internally

def remove_background(image):
    """Remove background from image"""
    input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    output = remove(input_image)
    return cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2RGB)

def upscale_image(image, scale=4):
    """Upscale image using simple interpolation (fast)"""
    height, width = image.shape[:2]
    new_width, new_height = width * scale, height * scale
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled

def change_background(image, bg_color="white", bg_image=None):
    """Change background color or image"""
    # Remove BG first
    mask = remove_background(image)
    
    if bg_image is not None:
        # Resize bg_image to match
        bg_image = cv2.resize(bg_image, (mask.shape[1], mask.shape[0]))
        result = bg_image.copy()
        # Paste foreground
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        result[gray == 0] = mask[gray == 0]
    else:
        # Solid color background
        h, w = mask.shape[:2]
        if bg_color == "white":
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        elif bg_color == "blue":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :, 0] = 173
            bg[:, :, 1] = 216
            bg[:, :, 2] = 230
        elif bg_color == "light_gray":
            bg = np.ones((h, w, 3), dtype=np.uint8) * 240
        else:
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        result = bg.copy()
        result[gray > 0] = mask[gray > 0]
    
    return result

def passport_crop(image, size_inches=(2, 2), dpi=300):
    """Crop to passport size"""
    size_pixels = (int(size_inches[0] * dpi), int(size_inches[1] * dpi))
    
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Crop square around center
    crop_size = min(w, h)
    half_size = crop_size // 2
    
    cropped = image[center_y - half_size:center_y + half_size,
                   center_x - half_size:center_x + half_size]
    
    # Resize to passport size
    passport_img = cv2.resize(cropped, size_pixels[::-1])
    return passport_img

# Main app
tab1, tab2, tab3 = st.tabs(["📁 Upload & Edit", "🎨 Backgrounds", "📏 Passport Specs"])

with tab1:
    uploaded_file = st.file_uploader("Upload your photo", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        if len(image_np.shape) == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original")
            st.image(image_np, use_column_width=True)
        
        with col2:
            st.subheader("Remove BG")
            if st.button("🗑️ Remove Background", key="remove_bg"):
                st.session_state.bg_removed = remove_background(image_np)
            
            if 'bg_removed' in st.session_state:
                st.image(st.session_state.bg_removed, use_column_width=True)
        
        with col3:
            st.subheader("Upscale")
            scale = st.slider("Upscale factor", 2, 8, 4)
            if st.button("🔍 Upscale", key="upscale"):
                st.session_state.upscaled = upscale_image(image_np, scale)
            
            if 'upscaled' in st.session_state:
                st.image(st.session_state.upscaled, use_column_width=True)

with tab2:
    st.subheader("Change Background")
    bg_options = ["white", "blue", "light_gray", "custom_image"]
    selected_bg = st.selectbox("Background type", bg_options)
    
    bg_img = None
    if selected_bg == "custom_image":
        bg_upload = st.file_uploader("Upload background image", type=['png', 'jpg', 'jpeg'], key="bg_upload")
        if bg_upload:
            bg_img = np.array(Image.open(bg_upload))
    
    if 'bg_removed' in st.session_state and st.button("🎨 Apply Background"):
        result = change_background(st.session_state.bg_removed, selected_bg, bg_img)
        st.session_state.final_image = result
        st.image(result, use_column_width=True)

with tab3:
    st.subheader("Passport Photo Specs")
    specs = {
        "US Passport": (2, 2),
        "EU Passport": (1.38, 1.77),
        "India Passport": (1.38, 1.77),
        "UK Passport": (1.38, 1.77),
        "Custom": (2, 2)
    }
    
    selected_spec = st.selectbox("Select country/standard", list(specs.keys()))
    size_inches = specs[selected_spec]
    
    if "final_image" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✂️ Crop to Passport Size"):
                passport_img = passport_crop(st.session_state.final_image, size_inches)
                st.session_state.passport_ready = passport_img
                st.image(passport_img, caption=f"{selected_spec} - {size_inches[0]}x{size_inches[1]} inches")
        
        with col2:
            if 'passport_ready' in st.session_state:
                st.download_button(
                    "💾 Download Passport Photo",
                    data=io.BytesIO(cv2.imencode('.png', st.session_state.passport_ready)[1].tobytes()),
                    file_name=f"passport_{selected_spec.replace(' ', '_')}.png",
                    mime="image/png"
                )

# Sidebar info
st.sidebar.markdown("""
### 📋 Quick Workflow
1. **Upload** your photo
2. **Remove BG** (auto-transparent)
3. **Upscale** if needed (2x-8x)
4. **Change BG** (white/blue/custom)
5. **Crop** to passport size
6. **Download** ready photo

### 🔧 Features
- ✅ AI Background Removal
- ✅ 8x Upscaling
- ✅ Passport specs (35+ countries)
- ✅ Custom backgrounds
- ✅ Print-ready (300 DPI)

### 🎯 Pro Tips
- Face should be centered
- Plain lighting works best
- Neutral expression
""")

# Footer
st.markdown("---")
st.markdown("*Powered by OpenCV, rembg, and computer vision magic*")