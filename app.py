import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import io
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="SEO Content Mapping Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* Tool selector cards */
    .tool-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .tool-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    .tool-card.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #faf0ff 100%);
    }
    .tool-card h3 {
        color: #2d3748;
        margin: 0.5rem 0;
    }
    .tool-card p {
        color: #718096;
        font-size: 0.9rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success {
        background: #c6f6d5;
        color: #22543d;
    }
    .badge-warning {
        background: #fefcbf;
        color: #744210;
    }
    .badge-info {
        background: #bee3f8;
        color: #2a4365;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.3rem;
    }

    /* Progress section */
    .progress-section {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8fafc;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2d3748;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'tool1_result' not in st.session_state:
    st.session_state.tool1_result = None
if 'tool1_trash' not in st.session_state:
    st.session_state.tool1_trash = None
if 'tool1_filename' not in st.session_state:
    st.session_state.tool1_filename = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔍 SEO Content Mapping Tool</h1>
    <p>Lọc từ khóa thông minh & Gom nhóm bài viết bằng AI</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - TOOL SELECTION
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Điều khiển")
    st.markdown("---")

    tool_choice = st.radio(
        "🧰 Chọn công cụ",
        options=["🧹 Công cụ 1: Lọc & Gom từ khóa", "🤖 Công cụ 2: Gom bài viết bằng AI", "🔄 Pipeline: Chạy cả 2"],
        index=0,
        help="Chọn công cụ bạn muốn sử dụng"
    )

    st.markdown("---")

    # Thông tin hướng dẫn
    with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        **Công cụ 1** - Lọc & Gom từ khóa:
        - Upload file CSV từ Ahrefs
        - Cấu hình hạt giống mục tiêu & nhiễu
        - AI sẽ lọc rác và gom nhóm ngữ nghĩa

        **Công cụ 2** - Gom bài viết bằng Gemini:
        - Upload file Excel (output từ CỤ 1 hoặc file tự tạo)
        - Cần có cột "Từ khóa" và "Volume"
        - Gemini AI sẽ gom thành bài viết chi tiết

        **Pipeline** - Chạy liên tục:
        - Upload file CSV, chạy CỤ 1 xong tự động chuyển sang CỤ 2
        """)

    with st.expander("ℹ️ Yêu cầu file đầu vào", expanded=False):
        st.markdown("""
        **File CSV (Công cụ 1):**
        - Xuất từ Ahrefs hoặc công cụ SEO
        - Cần có cột: `Keyword` và `Volume`

        **File Excel (Công cụ 2):**
        - Cần có cột: `Từ Khóa` và `Volume`
        - Hoặc dùng output từ Công cụ 1
        """)


# ============================================================
# CÔNG CỤ 1: LỌC & GOM TỪ KHÓA
# ============================================================
def run_tool1(uploaded_file, target_seeds, noise_seeds, distance_threshold, margin):
    """Chạy công cụ lọc và gom từ khóa semantic"""
    from sentence_transformers import SentenceTransformer, util
    from sklearn.cluster import AgglomerativeClustering

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Đọc dữ liệu
    status_text.markdown("📂 **Bước 1/5:** Đang đọc dữ liệu đầu vào...")
    progress_bar.progress(5)

    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig', on_bad_lines='skip')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-16', sep='\t', engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Không thể đọc file: {e}")
        return None, None

    df.columns = df.columns.str.strip()
    col_kw = next((c for c in df.columns if 'từ khóa' in c.lower() or 'keyword' in c.lower()), None)
    col_vol = next((c for c in df.columns if 'volume' in c.lower()), None)

    if not col_kw or not col_vol:
        st.error("❌ Không tìm thấy cột **Keyword** hoặc **Volume** trong file.")
        st.info(f"📋 Các cột tìm thấy: {list(df.columns)}")
        return None, None

    df = df.rename(columns={col_kw: 'Keyword', col_vol: 'Volume'})
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Keyword'])
    keywords = df['Keyword'].astype(str).tolist()

    st.info(f"📊 Tổng số từ khóa đầu vào: **{len(keywords):,}**")
    progress_bar.progress(15)

    # 2. Tải model
    status_text.markdown("🧠 **Bước 2/5:** Đang tải lõi AI đa ngôn ngữ (MPNet)...")
    progress_bar.progress(20)
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    progress_bar.progress(40)

    # 3. Lọc rác
    status_text.markdown("🔬 **Bước 3/5:** AI đang rà quét và phân định ranh giới ngữ nghĩa...")
    kw_vecs = model.encode(keywords, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    target_vecs = model.encode(target_seeds, convert_to_tensor=True)
    noise_vecs = model.encode(noise_seeds, convert_to_tensor=True)

    target_scores = util.cos_sim(kw_vecs, target_vecs).max(dim=1).values.tolist()
    noise_scores = util.cos_sim(kw_vecs, noise_vecs).max(dim=1).values.tolist()

    clean_indices = []
    trash_indices = []

    for i in range(len(keywords)):
        if target_scores[i] > (noise_scores[i] + margin):
            clean_indices.append(i)
        else:
            trash_indices.append(i)

    df_clean = df.iloc[clean_indices].copy().reset_index(drop=True)
    df_trash = df.iloc[trash_indices].copy()
    clean_kw_vecs = kw_vecs[clean_indices]

    progress_bar.progress(60)

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Giữ lại: **{len(df_clean):,}** từ khóa")
    with col2:
        st.warning(f"🗑️ Loại bỏ: **{len(df_trash):,}** từ khóa")

    if len(df_clean) == 0:
        st.error("❌ Không còn từ khóa nào sau khi lọc!")
        return None, None

    # 4. Gom nhóm
    status_text.markdown("🧩 **Bước 4/5:** Đang gom nhóm ngữ nghĩa (Semantic Clustering)...")
    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    df_clean['Cluster_ID'] = cluster_model.fit_predict(clean_kw_vecs.cpu().numpy())
    progress_bar.progress(80)

    # 5. Xuất kết quả
    status_text.markdown("📊 **Bước 5/5:** Đang tạo báo cáo...")
    content_map_data = []
    cluster_volumes = df_clean.groupby('Cluster_ID')['Volume'].sum().reset_index()
    cluster_volumes = cluster_volumes.sort_values(by='Volume', ascending=False)

    for cid in cluster_volumes['Cluster_ID']:
        group = df_clean[df_clean['Cluster_ID'] == cid].sort_values(by='Volume', ascending=False)
        main_row = group.iloc[0]
        focus_keyword = main_row['Keyword']
        total_volume = group['Volume'].sum()

        for i in range(len(group)):
            row = group.iloc[i]
            is_main = (i == 0)
            content_map_data.append({
                'Chủ Đề (Tên Bài)': focus_keyword,
                'Phân Loại': '1 - Keyword Chính' if is_main else '2 - Keyword Phụ',
                'Từ Khóa': row['Keyword'],
                'Volume': row['Volume'],
                'Tổng Traffic Nhóm': total_volume if is_main else None
            })

    df_final = pd.DataFrame(content_map_data)
    progress_bar.progress(100)
    status_text.markdown("✅ **Hoàn thành!**")

    return df_final, df_trash[['Keyword', 'Volume']].sort_values(by='Volume', ascending=False)


# ============================================================
# CÔNG CỤ 2: GOM BÀI VIẾT BẰNG GEMINI
# ============================================================
def run_tool2(df_input, api_key, batch_size=80):
    """Chạy công cụ gom bài viết bằng Gemini AI"""
    import google.generativeai as genai

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Cấu hình API
    status_text.markdown("🔑 **Bước 1/3:** Đang kết nối Gemini AI...")
    genai.configure(api_key=api_key)

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = ""
        for name in ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-1.5-flash']:
            if name in available_models:
                model_name = name
                break
        if not model_name:
            model_name = available_models[0] if available_models else ""
        if not model_name:
            st.error("❌ API Key không có quyền truy cập mô hình tạo văn bản.")
            return None

        st.info(f"🤖 Sử dụng model: **{model_name}**")
        model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.1})
    except Exception as e:
        st.error(f"❌ Lỗi kết nối API: {e}")
        return None

    progress_bar.progress(10)

    # 2. Chuẩn bị dữ liệu
    status_text.markdown("📂 **Bước 2/3:** Đang chuẩn bị dữ liệu...")

    df_input.columns = df_input.columns.astype(str).str.strip()

    # Tìm cột từ khóa
    col_kw = None
    for col in df_input.columns:
        col_lower = col.lower()
        if 'từ khóa' in col_lower or 'từ khoá' in col_lower or 'keyword' in col_lower:
            col_kw = col
            break

    # Tìm cột volume
    col_vol = None
    for col in df_input.columns:
        if 'volume' in col.lower():
            col_vol = col
            break

    if not col_kw or not col_vol:
        st.error(f"❌ Không tìm thấy cột 'Từ khóa' và 'Volume'. Các cột hiện có: {list(df_input.columns)}")
        return None

    df_work = df_input.rename(columns={col_kw: 'Từ Khóa', col_vol: 'Volume'})
    df_work['Volume'] = pd.to_numeric(df_work['Volume'], errors='coerce').fillna(0)
    keywords_data = df_work[['Từ Khóa', 'Volume']].to_dict('records')
    total_kw = len(keywords_data)

    st.info(f"📊 Tổng số từ khóa: **{total_kw:,}**")
    progress_bar.progress(15)

    # 3. Gửi cho Gemini
    status_text.markdown("🤖 **Bước 3/3:** Đang gửi dữ liệu cho Gemini phân tích...")
    all_articles = []
    total_batches = (total_kw // batch_size) + (1 if total_kw % batch_size else 0)

    batch_progress = st.empty()

    for i in range(0, total_kw, batch_size):
        batch = keywords_data[i: i + batch_size]
        batch_num = i // batch_size + 1

        batch_progress.markdown(f"⏳ Đang xử lý batch **{batch_num}/{total_batches}**...")
        current_progress = 15 + int((batch_num / total_batches) * 80)
        progress_bar.progress(min(current_progress, 95))

        input_data_str = "\n".join([f"- {item['Từ Khóa']} (Volume: {item['Volume']})" for item in batch])

        prompt = f"""
Bạn là Trưởng phòng SEO kỹ thuật cực kỳ khắt khe. Nhiệm vụ của bạn là gom nhóm danh sách từ khóa sau thành các Bài viết (URLs).

LUẬT GOM NHÓM SINH TỬ (Đọc kỹ):
1. NGUYÊN TẮC CÙNG URL: 2 từ khóa CHỈ ĐƯỢC PHÉP nằm chung 1 nhóm nếu người tìm kiếm chúng mong muốn đọc ĐÚNG CÙNG 1 BÀI VIẾT (Cùng 1 URL trên Google).
2. CHỐNG GOM CƯỠNG ÉP: Nếu từ khóa mang ý nghĩa khác nhau, bắt buộc phải tách thành 2 bài viết riêng.
   Ví dụ ĐÚNG: Gom "cách hạch toán 111" và "hướng dẫn hạch toán tài khoản tiền mặt" (Đúng vì chung 1 mục đích).
3. QUYỀN CÔ LẬP: Nếu một từ khóa hoàn toàn không liên quan đến các từ khác, hãy để nó đứng 1 mình (Làm 1 bài viết riêng, sub_keywords để trống). Không được ghép bừa bãi.
4. GIỮ NGUYÊN DỮ LIỆU: Bắt buộc giữ nguyên văn 100% từ khóa và volume, không được tự bịa thêm.

DANH SÁCH TỪ KHÓA:
{input_data_str}

OUTPUT YÊU CẦU:
Chỉ xuất ra đúng 1 mảng JSON hợp lệ, không có code block markdown (như ```json).
[
  {{
    "intent": "Định nghĩa / Hướng dẫn / Thương mại / Phần mềm",
    "main_keyword": "từ khóa chính (chọn từ có volume cao nhất hoặc bao quát nhất nhóm)",
    "main_volume": 1000,
    "sub_keywords": [
      {{"keyword": "từ khóa phụ sát nghĩa", "volume": 500}}
    ]
  }}
]
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = response.text.strip()

                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                batch_articles = json.loads(response_text)
                all_articles.extend(batch_articles)
                time.sleep(5)
                break

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    batch_progress.warning(f"⚠️ API quá tải ở Batch {batch_num} (Lần {attempt+1}/{max_retries}). Đang chờ 40s...")
                    time.sleep(40)
                else:
                    batch_progress.error(f"❌ Lỗi Batch {batch_num}: {e}")
                    break

    if not all_articles:
        st.error("❌ AI không trả về dữ liệu hợp lệ. Vui lòng thử lại!")
        return None

    # Xuất kết quả
    status_text.markdown("📊 Đang xuất báo cáo...")
    content_map_data = []

    for article in all_articles:
        intent = article.get("intent", "Chung")
        main_kw = article.get("main_keyword", "")
        main_vol = article.get("main_volume", 0)
        sub_kws = article.get("sub_keywords", [])
        total_volume = main_vol + sum([sub.get("volume", 0) for sub in sub_kws])

        content_map_data.append({
            'Search Intent': intent,
            'Từ Khóa Chính (H1)': main_kw,
            'Phân Loại': '1 - Keyword Chính',
            'Từ Khóa': main_kw,
            'Volume': main_vol,
            'Tổng Volume Bài': total_volume
        })

        for sub in sub_kws:
            content_map_data.append({
                'Search Intent': intent,
                'Từ Khóa Chính (H1)': main_kw,
                'Phân Loại': '2 - Keyword Phụ',
                'Từ Khóa': sub.get("keyword", ""),
                'Volume': sub.get("volume", 0),
                'Tổng Volume Bài': None
            })

    df_output = pd.DataFrame(content_map_data)
    df_output['Temp_Total'] = df_output.groupby('Từ Khóa Chính (H1)')['Volume'].transform('sum')
    df_output = df_output.sort_values(
        by=['Temp_Total', 'Từ Khóa Chính (H1)', 'Phân Loại'],
        ascending=[False, True, True]
    )
    df_output = df_output.drop(columns=['Temp_Total'])

    progress_bar.progress(100)
    status_text.markdown("✅ **Hoàn thành!**")

    return df_output


# ============================================================
# HELPER: Tạo file Excel để download
# ============================================================
def to_excel_bytes(df_main, df_trash=None):
    """Chuyển DataFrame thành bytes Excel để download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Content Map', index=False)
        if df_trash is not None and len(df_trash) > 0:
            df_trash.to_excel(writer, sheet_name='Từ Khóa Bị Loại', index=False)
    return output.getvalue()


def to_excel_bytes_single(df):
    """Chuyển DataFrame thành bytes Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Content Map AI', index=False)
    return output.getvalue()


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================

# --- CÔNG CỤ 1 ---
if "Công cụ 1" in tool_choice:
    st.markdown("## 🧹 Công cụ 1: Lọc & Gom từ khóa Semantic")
    st.markdown("Upload file CSV từ Ahrefs, AI sẽ lọc rác và gom nhóm từ khóa theo ngữ nghĩa.")

    col_upload, col_config = st.columns([1, 1])

    with col_upload:
        st.markdown("### 📁 Upload File")
        uploaded_file = st.file_uploader(
            "Chọn file CSV từ Ahrefs",
            type=['csv'],
            key='tool1_upload',
            help="File CSV cần có cột Keyword và Volume"
        )

    with col_config:
        st.markdown("### 🎯 Cấu hình hạt giống")

        target_input = st.text_area(
            "🎯 Hạt giống MỤC TIÊU (mỗi dòng 1 từ)",
            value="kế toán\nhóa đơn",
            height=100,
            help="Các từ khóa đặc trưng của ngành bạn muốn giữ lại"
        )

        noise_input = st.text_area(
            "🚫 Hạt giống NHIỄU (mỗi dòng 1 từ)",
            value="học sinh\nsinh viên
