import io
import os
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv

# PDFMiner
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# LLM
from langchain_groq import ChatGroq

st.set_page_config(page_title="Phân loại CV", page_icon="📄", layout="centered")

load_dotenv()
ENV_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.header("Thiết lập")
    groq_key_input = st.text_input(
        "GROQ_API_KEY",
        type="password",
        value=ENV_GROQ_KEY if ENV_GROQ_KEY else "",
        help="Đặt trong .env hoặc nhập trực tiếp tại đây"
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input.strip()

st.title("📄 Phân loại CV vào **lĩnh vực** phù hợp nhất")
st.caption("Upload file PDF và hệ thống sẽ trả về đúng **3 lĩnh vực** theo định dạng yêu cầu.")
CLASSIFY_PROMPT = """
Bạn là một nhà tuyển dụng chuyên nghiệp, có nhiệm vụ phân tích và phân loại hồ sơ ứng viên. 

Hãy đọc kỹ nội dung CV dưới đây và **chỉ trả về đúng 3 lĩnh vực phù hợp nhất** từ danh sách cho sẵn bên dưới, dựa trên kinh nghiệm, kỹ năng và định hướng nghề nghiệp của ứng viên.

**Yêu cầu quan trọng:**
- Chỉ liệt kê đúng 3 lĩnh vực phù hợp nhất.
- Không cần giải thích hay mô tả thêm.
- Chỉ trả về kết quả đúng theo định dạng sau:

**Kết quả:**
1. [Tên lĩnh vực 1]
2. [Tên lĩnh vực 2]
3. [Tên lĩnh vực 3]

**Danh sách lĩnh vực:**
1. Công nghệ - Thông tin
2. Đầu tư - Tài chính
3. Y tế - Dược phẩm
4. Giáo dục
5. Bất động sản - Xây dựng
6. Năng lượng - Môi trường
7. Thực phẩm - Nông nghiệp
8. Dịch vụ - Du lịch
9. Sản phẩm - Tiêu dùng
10. Nhà hàng - Ăn uống
11. Vận tải - Logistics
12. Thể dục - Thể thao

### Đây là nội dung CV:
--HERERESUME--
""".strip()


# ========== HÀM XỬ LÝ ==========
def load_pdf_text(uploaded_file) -> str:
    """
    Trích xuất văn bản từ PDF (Streamlit UploadedFile).
    """
    try:
        if uploaded_file is None:
            return ""

        # Đảm bảo đúng định dạng PDF
        if not uploaded_file.name.lower().endswith(".pdf"):
            raise ValueError("Vui lòng upload file PDF.")
        file_bytes = uploaded_file.getvalue()
        bio = io.BytesIO(file_bytes)

        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, converter)

        for page in PDFPage.get_pages(bio, caching=True, check_extractable=True):
            interpreter.process_page(page)

        text = fake_file_handle.getvalue()
        converter.close()
        fake_file_handle.close()
        bio.close()

        return text or ""
    except Exception as e:
        st.error(f"Không thể trích xuất nội dung từ PDF: {e}")
        return ""


def remove_line_breaks(text: str) -> str:
    return re.sub(r"\r\n|\r|\n", " ", text)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_upload_hash(uploaded_file) -> str:
    """
    Tạo hash dựa trên tên + kích thước + 512 bytes đầu để nhận biết file mới.
    """
    if not uploaded_file:
        return ""
    head = uploaded_file.getvalue()[:512]
    h = hashlib.sha256()
    h.update(uploaded_file.name.encode("utf-8"))
    h.update(str(uploaded_file.size).encode("utf-8"))
    h.update(head)
    return h.hexdigest()


def call_llm(resume_text: str) -> str:
    """
    Gọi Groq LLM qua langchain_groq, chỉ trả về content (string).
    """
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )
    messages = [
        ("system", "Bạn là một nhà tuyển dụng nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể."),
        ("human", CLASSIFY_PROMPT.replace("--HERERESUME--", resume_text)),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)


def parse_top3_fields(text: str):
    """
    Rút trích 3 dòng kết quả theo định dạng mong muốn.
    Chấp nhận các biến thể nhỏ (có/không có 'Kết quả:'), và đánh số 1./2./3.
    """
    # Thử tách theo block "Kết quả:" nếu có
    block = text
    m = re.search(r"Kết\s*quả\s*:?(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        block = m.group(1)

    # Lấy 3 dòng bắt đầu bằng số thứ tự
    lines = re.findall(r"^\s*\d+\.\s*(.+?)\s*$", block, flags=re.MULTILINE)
    lines = [l.strip(" -•\t") for l in lines if l.strip()]

    # Nếu không tìm thấy theo mẫu đánh số, cố gắng tách theo xuống dòng
    if len(lines) < 3:
        fallback = [s.strip() for s in block.splitlines() if s.strip()]
        lines = [l for l in fallback if l][:3]

    # Chỉ giữ 3 mục đầu
    return lines[:3]


# ========== QUẢN LÝ STATE ĐỂ XÓA KẾT QUẢ CŨ ==========
if "last_upload_hash" not in st.session_state:
    st.session_state.last_upload_hash = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ========== GIAO DIỆN ==========
uploaded_file = st.file_uploader("Tải lên CV (PDF)", type=["pdf"])

# Tạo hash cho file hiện tại
current_hash = get_upload_hash(uploaded_file)
if current_hash != st.session_state.last_upload_hash:
    # File mới -> xóa kết quả cũ
    st.session_state.last_result = None
    st.session_state.last_upload_hash = current_hash

run_btn_col1, run_btn_col2 = st.columns([1, 1])
with run_btn_col1:
    run_now = st.button("Phân loại ngay", type="primary", disabled=(uploaded_file is None))
with run_btn_col2:
    clear_now = st.button("Xóa kết quả")

if clear_now:
    st.session_state.last_result = None

if run_now:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Chưa có GROQ_API_KEY. Hãy đặt trong .env hoặc nhập ở sidebar.")
    elif uploaded_file is None:
        st.error("Vui lòng upload một file PDF.")
    else:
        with st.spinner("Đang trích xuất và phân loại…"):
            raw_text = load_pdf_text(uploaded_file)
            formatted_text = remove_line_breaks(raw_text)
            cleaned_text = clean_text(formatted_text)

            if not cleaned_text:
                st.error("Không đọc được nội dung CV từ file PDF.")
            else:
                try:
                    response_text = call_llm(cleaned_text)
                    top3 = parse_top3_fields(response_text)

                    # Nếu parse không được, hiển thị nguyên văn
                    if len(top3) != 3:
                        st.warning("Không trích xuất được đúng 3 mục, hiển thị nguyên văn phản hồi:")
                        st.code(response_text, language="markdown")
                    else:
                        st.session_state.last_result = top3
                except Exception as e:
                    st.error(f"Lỗi khi gọi mô hình: {e}")

st.markdown("---")
st.subheader("Kết quả")
if st.session_state.last_result:
    # Đúng định dạng mong muốn: chỉ 3 dòng
    st.markdown("**Kết quả:**")
    st.markdown(f"1. {st.session_state.last_result[0]}")
    st.markdown(f"2. {st.session_state.last_result[1]}")
    st.markdown(f"3. {st.session_state.last_result[2]}")
else:
    st.info("Chưa có kết quả. Hãy upload file PDF và bấm **Phân loại ngay**.")
