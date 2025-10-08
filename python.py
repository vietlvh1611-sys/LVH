import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import generation_types
from google.api_core import exceptions

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính với Gemini AI 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'. Vui lòng kiểm tra lại file Excel.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini ---
def get_gemini_response(api_key, prompt, is_chat=False, chat_history=None):
    """
    Gửi yêu cầu đến Gemini API.
    - is_chat=False: Cho yêu cầu phân tích ban đầu (one-shot).
    - is_chat=True: Cho cuộc trò chuyện, cần có chat_history.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        if is_chat:
            # Đối với chat, bắt đầu một session với lịch sử
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(prompt)
        else:
            # Đối với yêu cầu phân tích ban đầu
            response = model.generate_content(prompt)
        
        return response.text

    except exceptions.PermissionDenied:
        return "Lỗi gọi Gemini API: Khóa API không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại."
    except generation_types.StopCandidateException as e:
        # Xử lý trường hợp nội dung bị chặn do an toàn
        return f"Phản hồi từ AI đã bị chặn. Lý do: {e}. Vui lòng thử lại với câu hỏi khác."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi kết nối với Gemini: {e}"

# --- Giao diện chính ---

# Chức năng 1: Tải File
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (3 cột: Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file, header=None) # Đọc không có header
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột và đặt tên
        if df_raw.shape[1] < 3:
             st.error("Lỗi: File Excel phải có ít nhất 3 cột.")
        else:
            df_raw = df_raw.iloc[:, :3]
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            st.session_state.df_raw = df_raw.copy() # Lưu vào session state
            
            # Xử lý dữ liệu
            df_processed = process_financial_data(df_raw.copy())
            st.session_state.df_processed = df_processed # Lưu vào session state

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

if 'df_processed' in st.session_state:
    df_processed = st.session_state.df_processed
    
    # --- Chức năng 2 & 3: Hiển thị Kết quả ---
    st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
    st.dataframe(df_processed.style.format({
        'Năm trước': '{:,.0f}',
        'Năm sau': '{:,.0f}',
        'Tốc độ tăng trưởng (%)': '{:.2f}%',
        'Tỷ trọng Năm trước (%)': '{:.2f}%',
        'Tỷ trọng Năm sau (%)': '{:.2f}%'
    }), use_container_width=True)
    
    # --- Chức năng 4: Tính Chỉ số Tài chính ---
    st.subheader("4. Các Chỉ số Tài chính Cơ bản")
    try:
        tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
        tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
        no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
        no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

        thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else 0
        thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Chỉ số Thanh toán Hiện hành (Năm trước)", value=f"{thanh_toan_hien_hanh_N_1:.2f} lần")
        with col2:
            st.metric(label="Chỉ số Thanh toán Hiện hành (Năm sau)", value=f"{thanh_toan_hien_hanh_N:.2f} lần", delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
            
    except IndexError:
        st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
        thanh_toan_hien_hanh_N = "N/A"
        thanh_toan_hien_hanh_N_1 = "N/A"
    except ZeroDivisionError:
        st.warning("Nợ ngắn hạn bằng 0, không thể tính chỉ số thanh toán hiện hành.")
        thanh_toan_hien_hanh_N = "N/A"
        thanh_toan_hien_hanh_N_1 = "N/A"

    st.divider()

    # --- Lấy API Key ---
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    else:
        # Chuẩn bị dữ liệu để gửi cho AI
        data_for_ai_markdown = df_processed.to_markdown(index=False)
        st.session_state.data_for_ai = data_for_ai_markdown
        
        # --- Chức năng 5: Nhận xét AI ban đầu ---
        st.subheader("5. Nhận xét Tổng quan từ AI")
        if st.button("Tạo nhận xét tổng quan"):
            prompt_initial = f"""
            Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên bảng dữ liệu sau đây, hãy đưa ra một nhận xét khách quan, ngắn gọn (3-4 đoạn) về tình hình tài chính của doanh nghiệp. 
            Phân tích của bạn cần tập trung vào các điểm chính sau:
            1. Tốc độ tăng trưởng của các chỉ tiêu quan trọng (Tài sản, Nguồn vốn, Doanh thu, Lợi nhuận nếu có).
            2. Sự thay đổi trong cơ cấu tài sản và nguồn vốn (Tỷ trọng).
            3. Đánh giá sơ bộ về khả năng thanh toán hiện hành.
            
            Dữ liệu phân tích:
            {st.session_state.data_for_ai}
            """
            with st.spinner('Gemini đang phân tích...'):
                ai_result = get_gemini_response(api_key, prompt_initial)
                st.info(ai_result)
                st.session_state.initial_analysis = ai_result
        
        st.divider()

        # --- Chức năng 6: Khung Chat tương tác ---
        st.subheader("6. Trò chuyện với AI về dữ liệu")

        # Khởi tạo lịch sử chat
        if "messages" not in st.session_state:
            # Bắt đầu với một tin nhắn chào mừng từ AI
            st.session_state.messages = [{
                "role": "assistant", 
                "content": "Tôi đã sẵn sàng. Bạn muốn hỏi gì về báo cáo tài chính này?"
            }]

        # Hiển thị các tin nhắn đã có
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Lấy input từ người dùng
        if user_prompt := st.chat_input("Đặt câu hỏi của bạn ở đây..."):
            # Thêm tin nhắn của người dùng vào lịch sử và hiển thị
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Tạo prompt hoàn chỉnh cho AI
            # Ghép nối bối cảnh dữ liệu, phân tích ban đầu (nếu có) và câu hỏi mới
            context_prompt = f"""
            Bối cảnh: Bạn là một chuyên gia tài chính đang trò chuyện với người dùng về bảng dữ liệu sau:
            {st.session_state.data_for_ai}
            
            {"Phân tích tổng quan ban đầu của bạn là: " + st.session_state.initial_analysis if 'initial_analysis' in st.session_state else ""}

            Hãy dựa vào bối cảnh trên để trả lời câu hỏi của người dùng một cách ngắn gọn, chính xác.
            """
            
            # Chuẩn bị lịch sử chat cho API
            chat_history_for_api = []
            for msg in st.session_state.messages:
                chat_history_for_api.append({
                    "role": "model" if msg["role"] == "assistant" else msg["role"],
                    "parts": [msg["content"]]
                })

            # Lấy và hiển thị phản hồi của AI
            with st.chat_message("assistant"):
                with st.spinner("AI đang suy nghĩ..."):
                    full_prompt = context_prompt + "\n\nCâu hỏi của người dùng: " + user_prompt
                    response = get_gemini_response(api_key, user_prompt, is_chat=True, chat_history=chat_history_for_api[:-1])
                    st.markdown(response)
            
            # Thêm phản hồi của AI vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Chào mừng bạn đến với ứng dụng Phân tích Báo cáo Tài chính. Vui lòng tải lên file Excel để bắt đầu.")
