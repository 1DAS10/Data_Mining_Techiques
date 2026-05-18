# Hệ thống Khai phá Dữ liệu (Data Mining System)

Ứng dụng web trực quan dành cho khai phá và phân tích dữ liệu, được xây dựng bằng **Streamlit**. Hỗ trợ các thuật toán khai phá dữ liệu phổ biến như phân loại Naive Bayes, cây quyết định ID3, luật kết hợp Apriori, tương quan Pearson và lý thuyết tập thô.

---

## 🚀 Yêu cầu hệ thống

- **Python** >= 3.9
- **pip** (Python package manager)

## 📦 Cài đặt

### Bước 1: Clone repository

```bash
git clone <url-repository>
cd Project
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Bước 3: Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, ứng dụng sẽ tự động mở tại địa chỉ: **http://localhost:8501**

---

## 🧭 Hướng dẫn sử dụng

1. **Tải dữ liệu lên**: Chọn file CSV hoặc Excel ở sidebar bên trái.
2. **Chọn thuật toán**: Chọn công cụ phân tích từ menu dropdown.
3. **Cấu hình tham số**: Điều chỉnh tỉ lệ kiểm tra và độ sâu cây quyết định (nếu cần).
4. **Xem kết quả**: Kết quả phân tích hiển thị trực quan ngay trên trang chính.

### Các thuật toán hỗ trợ:

| Thuật toán | Mô tả |
|---|---|
| 📋 **Trình diễn dữ liệu** | Xem tổng quan, thống kê, biểu đồ phân bổ |
| 🔗 **Tương quan Pearson** | Tính hệ số tương quan giữa các thuộc tính |
| 🛒 **Luật kết hợp (Apriori)** | Tìm tập mục thường xuyên |
| 🕊️ **Phân loại Naive Bayes** | Phân loại dựa trên xác suất Bayes |
| 🌳 **Cây quyết định ID3** | Xây dựng cây quyết định với chỉ số Entropy |
| 🧮 **Lý thuyết Tập thô** | Xử lý dữ liệu không chắc chắn (Rough Set) |

---

## 📁 Cấu trúc thư mục

```
Project/
├── app.py                  # Ứng dụng Streamlit chính
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── data/                   # Thư mục chứa dữ liệu mẫu
│   └── play_tennis.csv
├── notebooks/              # Jupyter notebooks phân tích
│   ├── DataMining.py
│   └── DataMiningProject.ipynb
├── scripts/                # Scripts tiện ích
│   └── transform_dataset.py
└── templates/              # Templates HTML (nếu dùng Flask)
    └── index.html
```

---

## 🛠 Khắc phục sự cố

**Lỗi "Module not found"**: Chạy lại `pip install -r requirements.txt`.

**Lỗi port 8501 đã được sử dụng**: Chạy với cờ `--server.port`:
```bash
streamlit run app.py --server.port 8502
```
