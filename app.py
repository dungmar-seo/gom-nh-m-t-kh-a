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

# Tắt các cảnh báo để giao diện sạch sẽ
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="SEO Clustering Tool", layout="wide", page_icon="🚀")

# Giao diện CSS tùy chỉnh - Đã sửa lỗi unsafe_allow_html
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold; }
    .stDownloadButton>button { width: 100%; border-radius: 8px; background-color: #008CBA; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Hệ Thống SEO: Lọc & Gom Nhóm Thông Minh")
st.info("Quy trình: Lọc rác ngữ nghĩa (Tool 1) → Mapping Content chuyên sâu bằng AI (Tool 2)")

# Khởi tạo bộ nhớ tạm
if 'tool1_output_df' not in st.session_state:
    st.session_state.tool1_output_df = None

# ================= HÀM XỬ LÝ CÔNG CỤ 1 (Semantic Filter) =================
@st.cache_resource
def load_semantic_model():
    # Sử dụng bản MiniLM nhẹ để chạy mượt trên Cloud
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def tool1_semantic_clustering(df, target_seeds, noise_seeds):
    df.columns = df.columns.astype(str).str.strip()
    # Tìm cột Keyword linh hoạt hơn
    col_kw = next((c for c in df.columns if any(k in c.lower() for k in ['từ khóa', 'keyword', 'từ khoá', 'search query'])), None)
    # Tìm cột Volume linh hoạt hơn
    col_vol = next((c for c in df.columns if any(k in c.lower() for k in ['volume', 'search volume', 'lượng tìm kiếm'])), None)

    if not col_kw or not col_vol:
        st.error(f"❌ Không tìm thấy cột 'Từ khóa' hoặc 'Volume'. Các cột hiện có: {list(df.columns)}")
        return None, None

    df = df.rename(columns={col_kw: 'Keyword', col_vol: 'Volume'})
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Keyword'])
    keywords = df['Keyword'].astype(str).tolist()

    model = load_semantic_model()
    
    with st.spinner("AI đang lọc rác ngữ nghĩa..."):
        kw_vecs = model.encode(keywords, batch_size=32, convert_to_tensor=True)
        target_vecs = model.encode(target_seeds, convert_to_tensor=True)
        noise_vecs = model.encode(noise_seeds, convert_to_tensor=True)

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
        st.warning("Không có từ khóa nào phù hợp với ngành mục tiêu sau khi lọc.")
        return None, None

    with st.spinner("Đang gom nhóm sơ bộ..."):
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
    col_kw = next((c for c in df.columns if any(k in c.lower() for k in ['từ khóa', 'keyword', 'từ khoá'])), None)
    col_vol = next((c for c in df.columns if 'volume' in c.lower()), None)

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
        
        # Sửa lỗi Syntax: Nối chuỗi an toàn
        prompt = (
            "Hãy đóng vai chuyên gia SEO. Gom nhóm các từ khóa sau thành bài viết.\n"
            "Luật: Cùng mục đích tìm kiếm thì vào 1 nhóm. Không được sửa từ khóa.\n"
            f"Danh sách:\n{input_data_str}\n"
            "Trả về JSON array duy nhất: "
            '[{"intent": "Dạng bài", "main_keyword": "từ chính", "main_volume": 100, "sub_keywords": [{"keyword": "từ phụ", "volume": 10}]}]'
        )

        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                txt = response.text.strip()
                if "
