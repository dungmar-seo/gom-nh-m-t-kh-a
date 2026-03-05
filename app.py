import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import google.generativeai as genai
import json
import time
import io
import os
import warnings

# Tắt cảnh báo để giao diện sạch sẽ hơn
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="SEO Clustering Tool", layout="wide", page_icon="🚀")

# CSS tùy chỉnh để làm giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #008CBA; color: white; }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🚀 Hệ Thống SEO: Lọc & Gom Nhóm Thông Minh")
st.info("Công cụ tích hợp: Lọc rác ngữ nghĩa (Tool 1) và Mapping Content bằng Gemini AI (Tool 2)")

# Khởi tạo bộ nhớ tạm để chuyển dữ liệu giữa 2 công cụ
if 'tool1_output_df' not in st.session_state:
    st.session_state.tool1_output_df = None

# ================= HÀM XỬ LÝ CÔNG CỤ 1 (Semantic Filter) =================
@st.cache_resource
def load_semantic_model():
    # Sử dụng bản MiniLM nhẹ hơn để tránh lỗi RAM trên Streamlit Cloud
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def tool1_semantic_clustering(df, target_seeds, noise_seeds):
    # Chuẩn hóa tên cột
    df.columns = df.columns.astype(str).str.strip()
    col_kw = next((c for c in df.columns if 'từ khóa' in c.lower() or 'keyword' in c.lower()), None)
    col_vol = next((c for c in df.columns if 'volume' in c.lower()), None)

    if not col_kw or not col_vol:
        st.error("❌ Lỗi: File của bạn thiếu cột 'Keyword' hoặc 'Volume'.")
        return None, None

    df = df.rename(columns={col_kw: 'Keyword', col_vol: 'Volume'})
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Keyword'])
    keywords = df['Keyword'].astype(str).tolist()

    model = load_semantic_model()
    
    with st.spinner("AI đang phân tích ngữ nghĩa để loại bỏ từ khóa rác..."):
        kw_vecs = model.encode(keywords, batch_size=32, convert_to_tensor=True)
        target_vecs = model.encode(target_seeds, convert_to_tensor=True)
        noise_vecs = model.encode(noise_seeds, convert_to_tensor=True)

        # Tính toán độ tương đồng Cosine
        target_scores = util.cos_sim(kw_vecs, target_vecs).max(dim=1).values.tolist()
        noise_scores = util.cos_sim(kw_vecs, noise_vecs).max(dim=1).values.tolist()

        clean_indices, trash_indices = [], []
        margin = 0.05
        for i in range(len(keywords)):
            if target_scores[i] > (noise_scores[i] + margin):
                clean_indices.append(i)
            else:
                trash_indices.append(i)

        df_clean = df.iloc[clean_indices].copy().reset_index(drop=True)
        df_trash = df.iloc[trash_indices].copy()
        clean_kw_vecs = kw_vecs[clean_indices]

    if len(df_clean) == 0:
        st.warning("Kết quả trống: Không tìm thấy từ khóa nào khớp với chủ đề mục tiêu.")
        return None, None

    with st.spinner("Đang gom nhóm các từ khóa đồng nghĩa..."):
        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, metric='cosine', linkage='average')
        df_clean['Cluster_ID'] = cluster_model.fit_predict(clean_kw_vecs.cpu().numpy())

        content_map_data = []
        cluster_volumes = df_clean.groupby('Cluster_ID')['Volume'].sum().reset_index().sort_values(by='Volume', ascending=False)

        for cid in cluster_volumes['Cluster_ID']:
            group = df_clean[df_clean['Cluster_ID'] == cid].sort_values(by='Volume', ascending=False)
            focus_keyword = group.iloc[0]['Keyword']
            total_volume = group['Volume'].sum()

            for i in range(len(group)):
                row = group.iloc[i]
                content_map_data.append({
                    'Chủ Đề (Tên Bài)': focus_keyword,
                    'Phân Loại': '1 - Keyword Chính' if i == 0 else '2 - Keyword Phụ',
                    'Từ Khóa': row['Keyword'],
                    'Volume': row['Volume'],
                    'Tổng Traffic Nhóm': total_volume if i == 0 else None
                })

    return pd.DataFrame(content_map_data), df_trash

# ================= HÀM XỬ LÝ CÔNG CỤ 2 (Gemini AI) =================
def tool2_gemini_clustering(df, api_key):
    genai.configure(api_key=api_key)
    
    df.columns = df.columns.astype(str).str.strip()
    col_kw = next((c for c in df.columns if 'từ khóa' in c.lower() or 'keyword' in c.lower()), None)
    col_vol = next((c for c in df.columns if 'volume' in c.lower()), None)

    if not col_kw:
        st.error("Không tìm thấy cột chứa từ khóa.")
        return None

    df = df.rename(columns={col_kw: 'Từ Khóa', col_vol: 'Volume' if col_vol else 'Volume_Alt'})
    if 'Volume' not in df.columns: df['Volume'] = 0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    
    keywords_data = df[['Từ Khóa', 'Volume']].to_dict('records')
    BATCH_SIZE = 70 
    all_articles = []
    
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1})
    
    total_kw = len(keywords_data)
    total_batches = (total_kw // BATCH_SIZE) + 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_kw, BATCH_SIZE):
        batch = keywords_data[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        status_text.text(f"Gemini đang xử lý Batch {batch_num}/{total_batches}...")
        progress_bar.progress(batch_num / total_batches)

        input_data_str = "\n".join([f"- {item['Từ Khóa']} (Vol: {item['Volume']})" for item in batch])
        
        prompt = f"""Bạn là Trưởng phòng SEO. Gom nhóm từ khóa sau thành các Bài viết.
LUẬT:
1. Cùng URL: Chỉ gom nếu người dùng muốn đọc cùng 1 bài.
2. Không gom bừa: Khác ý nghĩa phải tách bài.
3. Giữ nguyên: Không sửa từ khóa, không bỏ Volume.

DANH SÁCH:
{input_data_str}

OUTPUT: JSON array duy nhất.
[
  {{
    "intent": "Định nghĩa/Hướng dẫn/Review",
    "main_keyword": "từ khóa chính",
    "main_volume": 1000,
    "sub_keywords": [{{"keyword": "từ phụ", "volume": 100}}]
  }}
]"""

        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                txt = response.text.strip()
                if "
