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
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def tool1_semantic_clustering(df, target_seeds, noise_seeds):
    df.columns = df.columns.astype(str).str.strip()
    col_kw = next((c for c in df.columns if any(k in c.lower() for k in ['từ khóa', 'keyword', 'từ khoá'])), None)
    col_vol = next((c for c in df.columns if 'volume' in c.lower()), None)

    if not col_kw or not col_vol:
        st.error("❌ Không tìm thấy cột 'Từ khóa' hoặc 'Volume'. Vui lòng kiểm tra lại file.")
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

        clean_indices = []
        trash_indices = []
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
        st.warning("Không có từ khóa nào phù hợp với ngành mục tiêu.")
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
        
        # Prompt gọn nhẹ để tránh lỗi ngắt quãng chuỗi
        prompt = "Hãy đóng vai chuyên gia SEO. Gom nhóm các từ khóa sau thành bài viết.\n"
        prompt += "Luật: Cùng mục đích tìm kiếm thì vào 1 nhóm. Không được sửa từ khóa.\n"
        prompt += f"Danh sách:\n{input_data_str}\n"
        prompt += "Trả về JSON array duy nhất: "
        prompt += '[{"intent": "Dạng bài", "main_keyword": "từ chính", "main_volume": 100, "sub_keywords": [{"keyword": "từ phụ", "volume": 10}]}]'

        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                txt = response.text.strip()
                if "```json" in txt:
                    txt = txt.split("```json")[1].split("```")[0]
                elif "```" in txt:
                    txt = txt.split("```")[1].split("```")[0]
                
                batch_articles = json.loads(txt.strip())
                all_articles.extend(batch_articles)
                time.sleep(4) 
                break
            except:
                time.sleep(5)
                    
    final_rows = []
    for art in all_articles:
        m_kw = art.get("main_keyword", "N/A")
        m_vol = art.get("main_volume", 0)
        intent = art.get("intent", "N/A")
        subs = art.get("sub_keywords", [])
        sub_total = sum([s.get("volume", 0) for s in subs])
        
        final_rows.append({
            'Intent': intent, 'Chủ đề chính': m_kw, 'Loại': 'Chính', 
            'Từ khóa': m_kw, 'Volume': m_vol, 'Tổng Vol Bài': m_vol + sub_total
        })
        for s in subs:
            final_rows.append({
                'Intent': intent, 'Chủ đề chính': m_kw, 'Loại': 'Phụ', 
                'Từ khóa': s.get("keyword", ""), 'Volume': s.get("volume", 0), 'Tổng Vol Bài': None
            })
            
    return pd.DataFrame(final_rows)

# ================= GIAO DIỆN CHÍNH =================
st.sidebar.title("Cấu hình công cụ")
mode = st.sidebar.selectbox("Chọn chức năng:", ["Tool 1: Lọc rác & Gom sơ bộ", "Tool 2: Gemini Mapping"])

st.sidebar.markdown("---")
user_api = st.sidebar.text_input("Nhập Gemini API Key:", type="password")
api_to_use = user_api if user_api else "AIzaSyBSPo-XImF7uXzZxpRTclt6-hSRxuS-U5g"

if mode == "Tool 1: Lọc rác & Gom sơ bộ":
    st.subheader("1️⃣ Lọc từ khóa & Loại bỏ rác ngữ nghĩa")
    c1, c2 = st.columns(2)
    with c1: t_seeds = st.text_area("Hạt giống NGÀNH (Mục tiêu):", "kế toán, hóa đơn")
    with c2: n_seeds = st.text_area("Hạt giống RÁC (Loại bỏ):", "học sinh, giải trí")
    
    file1 = st.file_uploader("Tải lên file từ khóa (CSV/Excel):", type=['csv', 'xlsx'])
    if file1 and st.button("Bắt đầu chạy Tool 1"):
        try:
            df_raw = pd.read_csv(file1) if file1.name.endswith('.csv') else pd.read_excel(file1)
            res_clean, res_trash = tool1_semantic_clustering(df_raw, t_seeds.split(","), n_seeds.split(","))
            if res_clean is not None:
                st.session_state.tool1_output_df = res_clean
                st.success("Xử lý xong! Dữ liệu đã được lưu vào bộ nhớ tạm.")
                out1 = io.BytesIO()
                res_clean.to_excel(out1, index=False)
                st.download_button("📥 Tải kết quả Tool 1", out1.getvalue(), "Ket_qua_Tool1.xlsx")
        except Exception as e:
            st.error(f"Lỗi: {e}")

elif mode == "Tool 2: Gemini Mapping":
    st.subheader("2️⃣ Gom nhóm chuyên sâu bằng AI")
    input_source = st.radio("Nguồn dữ liệu:", ["Kế thừa từ Tool 1", "Tải file mới"])
    
    df_to_process = None
    if input_source == "Kế thừa từ Tool 1":
        df_to_process = st.session_state.tool1_output_df
        if df_to_process is None: st.warning("Hãy chạy Tool 1 trước.")
    else:
        file2 = st.file_uploader("Tải file từ khóa mới:", type=['csv', 'xlsx'])
        if file2: df_to_process = pd.read_csv(file2) if file2.name.endswith('.csv') else pd.read_excel(file2)

    if df_to_process is not None and st.button("Chạy Gemini Mapping"):
        try:
            final_res = tool2_gemini_clustering(df_to_process, api_to_use)
            if final_res is not None:
                st.dataframe(final_res.head(20))
                out2 = io.BytesIO()
                final_res.to_excel(out2, index=False)
                st.download_button("📥 Tải Content Map", out2.getvalue(), "Final_Content_Map.xlsx")
        except Exception as e:
            st.error(f"Lỗi AI: {e}")
