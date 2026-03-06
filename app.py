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

# Tắt các cảnh báo hệ thống để giao diện sạch sẽ
warnings.filterwarnings("ignore")

# ================= 1. CẤU HÌNH GIAO DIỆN & STYLE =================
st.set_page_config(page_title="Hệ thống SEO AI Pro", layout="wide", page_icon="🚀")

# CSS tùy chỉnh để làm giao diện chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #FF4B4B; color: white; font-weight: bold; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #ff3333; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .stDownloadButton>button { width: 100%; border-radius: 10px; background-color: #28a745; color: white; }
    .logic-container { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 8px solid #FF4B4B; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .guide-title { color: #FF4B4B; font-size: 1.3em; font-weight: bold; margin-bottom: 10px; display: flex; align-items: center; }
    .step-badge { background-color: #FF4B4B; color: white; border-radius: 50%; width: 25px; height: 25px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. QUẢN LÝ TRẠNG THÁI (SESSION STATE) =================
# Cầu nối dữ liệu: Tool 1 lưu vào đây, Tool 2 lấy ra dùng mà không cần upload lại
if 'df_bridge' not in st.session_state:
    st.session_state.df_bridge = None

# ================= 3. SIDEBAR: CẤU HÌNH & ĐIỀU KHIỂN =================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=80)
st.sidebar.title("Hệ Thống SEO AI")
app_mode = st.sidebar.radio("Chọn chức năng làm việc:", 
    ["Công cụ 1: Lọc rác & Gom nhóm MPNet", "Công cụ 2: Mapping Intent bằng Gemini"])

st.sidebar.divider()
st.sidebar.subheader("🔑 Cấu hình Gemini API")
user_api_key = st.sidebar.text_input("Nhập API Key cá nhân:", type="password", help="Lấy key tại Google AI Studio")
# Key mặc định từ file aaaaa.txt của bạn
DEFAULT_API_KEY = "AIzaSyBSPo-XImF7uXzZxpRTclt6-hSRxuS-U5g"
final_api_key = user_api_key if user_api_key else DEFAULT_API_KEY

st.sidebar.info("""
**💡 Luồng làm việc tối ưu:**
1. Chạy **Tool 1** để AI lọc sạch rác ngữ nghĩa và gom nhóm sơ bộ.
2. Sang **Tool 2**, chọn 'Kế thừa' để Gemini lập bản đồ nội dung chuyên sâu (Content Map).
""")

# ================= 4. LOGIC CÔNG CỤ 1 (DỰA TRÊN BBBBBB.TXT) =================
@st.cache_resource
def load_mpnet_model():
    # Sử dụng model MPNet cực kỳ mạnh mẽ cho đa ngôn ngữ (Tiếng Việt rất tốt)
    # Đây là model thông minh nhất trong bbbbbb.txt
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def run_tool_1_logic(df, target_seeds, noise_seeds):
    # Chuẩn hóa tên cột để không bị lỗi dấu hoặc khoảng trắng
    df.columns = df.columns.astype(str).str.strip()
    col_kw = next((c for c in df.columns if any(k in c.lower() for k in ['từ khóa', 'keyword', 'từ khoá'])), None)
    col_vol = next((c for c in df.columns if any(k in c.lower() for k in ['volume', 'lượng tìm kiếm'])), None)

    if not col_kw or not col_vol:
        st.error(f"❌ Không tìm thấy cột Keyword/Volume. Cột hiện có: {list(df.columns)}")
        return None, None

    df = df.rename(columns={col_kw: 'Keyword', col_vol: 'Volume'})
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    keywords = df['Keyword'].astype(str).tolist()

    model = load_mpnet_model()
    
    with st.spinner("🧠 AI đang rà quét và phân định ranh giới ngữ nghĩa (Lọc rác)..."):
        # Chuyển từ khóa thành tọa độ Vector
        kw_vecs = model.encode(keywords, batch_size=64, convert_to_tensor=True)
        target_vecs = model.encode(target_seeds, convert_to_tensor=True)
        noise_vecs = model.encode(noise_seeds, convert_to_tensor=True)

        # Thuật toán Max-Score so sánh đối trọng giữa Ngành và Rác từ bbbbbb.txt
        target_scores = util.cos_sim(kw_vecs, target_vecs).max(dim=1).values.tolist()
        noise_scores = util.cos_sim(kw_vecs, noise_vecs).max(dim=1).values.tolist()

        clean_indices, trash_indices = [], []
        margin = 0.05 # Biên độ an toàn từ bbbbbb.txt
        for i in range(len(keywords)):
            if target_scores[i] > (noise_scores[i] + margin):
                clean_indices.append(i)
            else:
                trash_indices.append(i)

        df_clean = df.iloc[clean_indices].copy().reset_index(drop=True)
        df_trash = df.iloc[trash_indices].copy()
        clean_kw_vecs = kw_vecs[clean_indices]

    if len(df_clean) == 0:
        return None, df_trash

    with st.spinner("🖇️ Đang gom nhóm sơ bộ (Semantic Clustering)..."):
        # Sử dụng AgglomerativeClustering với ngưỡng 0.35 (đúng như bbbbbb.txt)
        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, metric='cosine', linkage='average')
        df_clean['Cluster_ID'] = cluster_model.fit_predict(clean_kw_vecs.cpu().numpy())

        # Tạo Content Map dữ liệu sơ bộ
        final_data = []
        cluster_volumes = df_clean.groupby('Cluster_ID')['Volume'].sum().reset_index().sort_values(by='Volume', ascending=False)

        for cid in cluster_volumes['Cluster_ID']:
            group = df_clean[df_clean['Cluster_ID'] == cid].sort_values(by='Volume', ascending=False)
            focus_kw = group.iloc[0]['Keyword']
            total_vol = group['Volume'].sum()

            for i in range(len(group)):
                final_data.append({
                    'Chủ Đề (Tên Bài)': focus_kw,
                    'Phân Loại': '1 - Keyword Chính' if i == 0 else '2 - Keyword Phụ',
                    'Từ Khóa': group.iloc[i]['Keyword'],
                    'Volume': group.iloc[i]['Volume'],
                    'Tổng Traffic Nhóm': total_vol if i == 0 else None
                })
    
    return pd.DataFrame(final_data), df_trash

# ================= 5. LOGIC CÔNG CỤ 2 (DỰA TRÊN AAAAA.TXT) =================
def run_tool_2_logic(df, api_key):
    genai.configure(api_key=api_key)
    
    df.columns = df.columns.astype(str).str.strip()
    col_kw = next((c for c in df.columns if any(k in c.lower() for k in ['từ khóa', 'keyword', 'từ khoá'])), None)
    col_vol = next((c for c in df.columns if any(k in c.lower() for k in ['volume', 'lượng tìm kiếm'])), None)

    df = df.rename(columns={col_kw: 'Từ Khóa', col_vol: 'Volume'})
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    keywords_data = df[['Từ Khóa', 'Volume']].to_dict('records')
    
    BATCH_SIZE = 80 # Batch size tối ưu từ aaaaa.txt
    total_kw = len(keywords_data)
    all_articles = []
    
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"temperature": 0.1})
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_kw, BATCH_SIZE):
        batch = keywords_data[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_kw // BATCH_SIZE) + 1
        status_text.text(f"🚀 Gemini đang xử lý Batch {batch_num}/{total_batches}...")
        progress_bar.progress(batch_num / total_batches)

        input_str = "\n".join([f"- {item['Từ Khóa']} (Volume: {item['Volume']})" for item in batch])
        
        # PROMPT GỐC TỪ AAAAA.TXT (LUẬT SINH TỬ)
        # Sử dụng cấu trúc nối chuỗi (f-string) cẩn thận để tránh lỗi unterminated string
        prompt = (
            "Bạn là Trưởng phòng SEO kỹ thuật cực kỳ khắt khe. Nhiệm vụ của bạn là gom nhóm danh sách từ khóa sau thành các Bài viết (URLs).\n\n"
            "LUẬT GOM NHÓM SINH TỬ (Đọc kỹ):\n"
            "1. NGUYÊN TẮC CÙNG URL: 2 từ khóa CHỈ ĐƯỢC PHÉP nằm chung 1 nhóm nếu người tìm kiếm chúng mong muốn đọc ĐÚNG CÙNG 1 BÀI VIẾT (Cùng 1 URL trên Google).\n"
            "2. CHỐNG GOM CƯỠNG ÉP: Nếu từ khóa mang ý nghĩa khác nhau, bắt buộc phải tách thành 2 bài viết riêng.\n"
            "3. QUYỀN CÔ LẬP: Nếu một từ khóa hoàn toàn không liên quan đến các từ khác, hãy để nó đứng 1 mình (Làm 1 bài viết riêng, sub_keywords để trống).\n"
            "4. GIỮ NGUYÊN DỮ LIỆU: Bắt buộc giữ nguyên văn 100% từ khóa và volume, không được tự bịa thêm.\n\n"
            f"DANH SÁCH TỪ KHÓA:\n{input_str}\n\n"
            "OUTPUT YÊU CẦU:\n"
            "Chỉ xuất ra đúng 1 mảng JSON hợp lệ, không có code block markdown.\n"
            "[\n"
            "  {\n"
            '    "intent": "Định nghĩa / Hướng dẫn / Thương mại / Phần mềm",\n'
            '    "main_keyword": "từ khóa chính",\n'
            '    "main_volume": 1000,\n'
            '    "sub_keywords": [\n'
            '      {"keyword": "từ khóa phụ sát nghĩa", "volume": 500}\n'
            "    ]\n"
            "  }\n"
            "]"
        )

        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                res_text = response.text.strip()
                # Làm sạch Markdown nếu AI tự ý thêm vào
                if res_text.startswith("
