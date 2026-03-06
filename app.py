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
st.set_page_config(page_title="CÔNG CỤ GOM NHÓM TỪ KHÓA - By Dũng MAR", layout="wide", page_icon="🚀")

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
st.sidebar.title("CÔNG CỤ GOM NHÓM TỪ KHÓA - By Dũng MAR")
app_mode = st.sidebar.radio("Chọn công cụ làm việc:", 
    ["Công cụ 1: Lọc từ khóa rác và gom nhóm từ khóa cơ bản từ File Ahrefs", "Công cụ 2: Phân nhóm từ khóa thành bài viết cụ thể bằng API Gemini"])

st.sidebar.divider()
st.sidebar.subheader("🔑 Cấu hình Gemini API")
user_api_key = st.sidebar.text_input("Nhập API Key của bạn để sử dụng Tool 2 vào đây (nếu không có mặc định sẽ có API miễn phí nhưng dễ bị quá tải gây lỗi):", type="password", help="Lấy key tại Google AI Studio")
# Key mặc định từ file aaaaa.txt của bạn
DEFAULT_API_KEY = "AIzaSyBSPo-XImF7uXzZxpRTclt6-hSRxuS-U5g"
final_api_key = user_api_key if user_api_key else DEFAULT_API_KEY

st.sidebar.info("""
**💡 Luồng thực hiện gom nhóm từ khóa tối ưu nhất:**
1. Chạy **Tool 1** để AI lọc sạch rác ngữ nghĩa và gom nhóm sơ bộ. Bạn có thể tải file này về để thực hiện chỉnh sửa bằng tay thêm 1 chút để đảm bảo từ khóa chỉ bao gồm từ khóa mong muốn.
2. Sang **Tool 2**, chọn 'Kế thừa' để Gemini lập bản đồ nội dung chuyên sâu (Content Map). Hoặc tải file từ khóa bạn muốn gom thành từng bài viết vào đây
""")

# ================= 4. LOGIC CÔNG CỤ 1 (DỰA TRÊN BBBBBB.TXT) =================
@st.cache_resource
def load_mpnet_model():
    # Sử dụng model MPNet từ bbbbbb.txt để đảm bảo độ chính xác cao nhất
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def run_tool_1_logic(df, target_seeds, noise_seeds):
    # Chuẩn hóa tên cột
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

        # Tạo Content Map dữ liệu sơ bộ theo cấu trúc bbbbbb.txt
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
                if res_text.startswith("```json"):
                    res_text = res_text[7:]
                elif res_text.startswith("```"):
                    res_text = res_text[3:]
                
                if res_text.endswith("```"):
                    res_text = res_text[:-3]
                
                batch_json = json.loads(res_text.strip())
                all_articles.extend(batch_json)
                time.sleep(5) # Nghỉ 5s chống rate limit
                break
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    status_text.warning(f"⚠️ Batch {batch_num} bị Rate Limit. Đang nghỉ giải lao 40 giây...")
                    time.sleep(40)
                else:
                    status_text.error(f"❌ Batch {batch_num} gặp lỗi: {str(e)}")
                    time.sleep(5)

    # Xử lý báo cáo như aaaaa.txt
    content_map_data = []
    for article in all_articles:
        intent = article.get("intent", "Chung")
        main_kw = article.get("main_keyword", "")
        main_vol = article.get("main_volume", 0)
        sub_kws = article.get("sub_keywords", [])
        total_v = main_vol + sum([s.get("volume", 0) for s in sub_kws])

        content_map_data.append({
            'Search Intent': intent, 'Từ Khóa Chính (H1)': main_kw, 'Phân Loại': '1 - Keyword Chính',
            'Từ Khóa': main_kw, 'Volume': main_vol, 'Tổng Volume Bài': total_v
        })
        for sub in sub_kws:
            content_map_data.append({
                'Search Intent': intent, 'Từ Khóa Chính (H1)': main_kw, 'Phân Loại': '2 - Keyword Phụ',
                'Từ Khóa': sub.get("keyword", ""), 'Volume': sub.get("volume", 0), 'Tổng Volume Bài': None
            })
            
    df_out = pd.DataFrame(content_map_data)
    df_out['Temp_Total'] = df_out.groupby('Từ Khóa Chính (H1)')['Volume'].transform('sum')
    df_out = df_out.sort_values(by=['Temp_Total', 'Từ Khóa Chính (H1)', 'Phân Loại'], ascending=[False, True, True])
    return df_out.drop(columns=['Temp_Total'])

# ================= 6. GIAO DIỆN CHÍNH (MAIN UI) =================
st.title("Hệ Thống SEO AI Pro 🚀")

if app_mode == "Công cụ 1: Lọc rác & Gom nhóm MPNet":
    st.header("🔍 Công cụ 1: Lọc Rác Ngữ Nghĩa (MPNet Base V2)")
    
    with st.expander("📖 Giải thích Logic & Hướng dẫn sử dụng", expanded=True):
        st.markdown(f"""
        <div class="logic-container">
            <div class="guide-title"><span class="step-badge">1</span> Cách thức hoạt động</div>
            Sử dụng lõi AI <b>MPNet Base V2</b> để chuyển từ khóa thành Vector ý nghĩa. Hệ thống so sánh từ khóa của bạn với nhóm từ khóa mục tiêu (Ngành) và nhóm rác.<br>
            - Nếu từ khóa giống nhóm từ khóa "Rác" hơn nhóm "Ngành", nó sẽ bị loại bỏ.<br>
            - Những từ khóa còn lại được gom nhóm bằng thuật toán máy học <i>Agglomerative Clustering</i>.<br><br>
            <div class="guide-title"><span class="step-badge">2</span> Cách nhập liệu tối ưu</div>
            - <b>Nhóm từ khóa đích:</b> Nhập các từ khóa "trụ cột" của sản phẩm (Ví dụ: <i>kế toán, hóa đơn, phần mềm</i>).<br>
            - <b>Nhóm từ khóa Rác:</b> Nhập các chủ đề dễ gây nhầm lẫn (Ví dụ: <i>phim ảnh, giải trí, nông nghiệp</i>).
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        target_in = st.text_area("Nhóm từ khóa đích (Target Seeds):", "kế toán, hóa đơn")
    with c2:
        noise_in = st.text_area("Nhóm từ khóa đích (Noise Seeds):", "học sinh, sinh viên, giải trí, mạng xã hội, facebook, game, phim ảnh, lừa đảo, tải nhạc, mua sắm")

    file1 = st.file_uploader("Tải file từ khóa gốc (Ahrefs CSV/Excel):", type=['csv', 'xlsx'])
    
    if file1 and st.button("🚀 Chạy Lọc & Gom Nhóm"):
        try:
            if file1.name.endswith('.csv'):
                # Xử lý Encoding đa dạng (Sửa lỗi 'utf-8 codec can't decode')
                encodings = ['utf-8-sig', 'utf-16', 'latin1', 'cp1252']
                df_in = None
                for enc in encodings:
                    try: 
                        file1.seek(0)
                        df_in = pd.read_csv(file1, encoding=enc)
                        break
                    except: continue
                if df_in is None: st.error("❌ Không thể đọc file CSV. Vui lòng kiểm tra lại định dạng.")
            else:
                df_in = pd.read_excel(file1)
            
            if df_in is not None:
                res_clean, res_trash = run_tool_1_logic(df_in, target_in.split(","), noise_in.split(","))
                
                if res_clean is not None:
                    st.session_state.df_bridge = res_clean # Lưu bridge cho Tool 2
                    st.success(f"✅ Hoàn thành! Đã giữ lại {len(res_clean)} từ khóa chất lượng.")
                    st.dataframe(res_clean.head(20))
                    
                    output_buf = io.BytesIO()
                    with pd.ExcelWriter(output_buf, engine='openpyxl') as writer:
                        res_clean.to_excel(writer, sheet_name='Content Map', index=False)
                        res_trash.to_excel(writer, sheet_name='Từ Khóa Bị Loại', index=False)
                    st.download_button("📥 Tải kết quả Công cụ 1", output_buf.getvalue(), "SEO_Tool1_Semantic.xlsx")
        except Exception as e: st.error(f"Lỗi: {str(e)}")

else:
    st.header("🧠 Công cụ 2: Mapping Intent bằng Gemini AI")
    
    with st.expander("📖 Giải thích Logic & Hướng dẫn sử dụng", expanded=True):
        st.markdown(f"""
        <div class="logic-container">
            <div class="guide-title"><span class="step-badge">1</span> Cách thức hoạt động</div>
            Sử dụng trí tuệ nhân tạo <b>Gemini 1.5 Flash</b> để hiểu <b>Search Intent</b> (ý định tìm kiếm).<br>
            - Hiểu sâu từ khóa để quyết định gom từ khóa vào chung một URL hay tách riêng.<br>
            - Phân loại chính xác Keyword chính (H1) và các Keyword phụ hỗ trợ.<br><br>
            <div class="guide-title"><span class="step-badge">2</span> Nguồn dữ liệu</div>
            Nên chọn <b>"Kế thừa từ Công cụ 1"</b> để xử lý dữ liệu đã được lọc sạch rác.
        </div>
        """, unsafe_allow_html=True)

    source = st.radio("Chọn nguồn dữ liệu đầu vào:", ["Kế thừa từ Công cụ 1", "Tải lên file mới"])
    
    df_to_ai = None
    if source == "Kế thừa từ Công cụ 1":
        if st.session_state.df_bridge is not None:
            st.info(f"📍 Đã nạp {len(st.session_state.df_bridge)} từ khóa đã lọc sạch từ Công cụ 1.")
            df_to_ai = st.session_state.df_bridge
        else:
            st.warning("⚠️ Chưa có dữ liệu từ Công cụ 1. Vui lòng quay lại bước 1 hoặc tải file mới.")
    else:
        file2 = st.file_uploader("Tải file từ khóa đã lọc sạch (Cần có cột 'Từ khóa' và 'Volume'):", type=['csv', 'xlsx'])
        if file2:
            if file2.name.endswith('.csv'):
                for enc in ['utf-8-sig', 'utf-16']:
                    try: file2.seek(0); df_to_ai = pd.read_csv(file2, encoding=enc); break
                    except: continue
            else: df_to_ai = pd.read_excel(file2)

    if df_to_ai is not None and st.button("🔥 Chạy AI Mapping Chuyên Sâu"):
        try:
            map_res = run_tool_2_logic(df_to_ai, final_api_key)
            if map_res is not None:
                st.success("🎉 Đã hoàn thành lập Content Map bằng AI!")
                st.dataframe(map_res.head(30))
                
                output_buf2 = io.BytesIO()
                map_res.to_excel(output_buf2, index=False)
                st.download_button("📥 Tải Content Map Cuối Cùng", output_buf2.getvalue(), "Final_SEO_Content_Map.xlsx")
        except Exception as e: st.error(f"Lỗi AI: {str(e)}")
