import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# 1. KHỞI TẠO LỊCH SỬ CHAT (Sử dụng session state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Nếu không tìm thấy, cố gắng tính tỷ trọng dựa trên tổng cột "Năm sau" và "Năm trước"
        # Điều này ít chính xác hơn nhưng đảm bảo app không crash
        tong_tai_san_N_1 = df['Năm trước'].sum()
        tong_tai_san_N = df['Năm sau'].sum()
        st.warning("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'. Tỷ trọng được tính dựa trên Tổng cộng các dòng dữ liệu.")
    else:
        tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
        tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân Tích Cố Định (Chức năng 5 gốc) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét theo prompt cố định."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.

        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM GỌI API GEMINI CHO CHAT (Chức năng mới) ---
def generate_chat_response(query, chat_history, financial_data_markdown, api_key):
    """Gửi query, lịch sử chat và dữ liệu tài chính làm ngữ cảnh cho Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Thiết lập System Instruction để định vị vai trò và ngữ cảnh
        system_instruction = f"""
        Bạn là một Trợ lý Phân tích Tài chính chuyên nghiệp và am hiểu về dữ liệu đã cho. Tên bạn là Gemini.
        Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng một cách chính xác, chuyên nghiệp và ngắn gọn.
        Bạn PHẢI sử dụng dữ liệu từ Báo cáo Tài chính hiện tại sau để trả lời (nếu có liên quan).
        Dữ liệu tài chính hiện tại (đã xử lý):
        {financial_data_markdown}
        """
        
        # Định dạng lịch sử chat thành định dạng API (role: user/model)
        contents = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in chat_history]
        
        # Thêm câu hỏi hiện tại của người dùng
        contents.append({"role": "user", "parts": [{"text": query}]})

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            system_instruction=system_instruction
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình chat: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    else:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            # Xử lý dữ liệu
            df_processed = process_financial_data(df_raw.copy())

            # Tính Chỉ số Thanh toán Hiện hành (dùng cho cả Phân tích Cố định và Context Chat)
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn & Nợ ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
            except IndexError:
                pass # Bỏ qua nếu thiếu chỉ tiêu, giá trị mặc định là "N/A"
            
            
            # Chuẩn bị dữ liệu Context cho AI (Cả cố định và Chat)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (Năm trước)', 
                    'Thanh toán hiện hành (Năm sau)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False), 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
                ]
            }).to_markdown(index=False)
            
            # --- TẠO TABS CHO GIAO DIỆN ---
            tab1, tab2 = st.tabs(["📊 Phân Tích Cơ Bản (Fixed Analysis)", "💬 Hỏi Đáp AI (Chat)"])

            with tab1:
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"
                    )
                with col2:
                    current_value = thanh_toan_hien_hanh_N if isinstance(thanh_toan_hien_hanh_N, float) else 0
                    previous_value = thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N_1, float) else 0
                    
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{current_value:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A",
                        delta=f"{current_value - previous_value:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    )
                
                if thanh_toan_hien_hanh_N == "N/A":
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")

                # --- Chức năng 5: Nhận xét AI Cố định ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                
                if st.button("Yêu cầu AI Phân tích Cố định"):
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
            
            with tab2:
                st.subheader("Trò chuyện với Gemini - Trợ lý Tài chính")
                st.info("Hãy hỏi tôi về bất kỳ chỉ tiêu nào trong bảng phân tích, tốc độ tăng trưởng, hoặc tỷ trọng cơ cấu tài sản.")

                # Hiển thị lịch sử chat
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Xử lý input mới từ người dùng
                if prompt := st.chat_input("Hỏi tôi về các chỉ tiêu tài chính..."):
                    
                    # Thêm prompt của người dùng vào lịch sử
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Tạo phản hồi từ Gemini
                    with st.chat_message("model"):
                        with st.spinner("Đang xử lý câu hỏi..."):
                            response = generate_chat_response(prompt, st.session_state.messages, data_for_ai, api_key)
                            
                            st.markdown(response)
                            # Thêm phản hồi của model vào lịch sử
                            st.session_state.messages.append({"role": "model", "content": response})


        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
