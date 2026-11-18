# LAB 09 — Named Entity Recognition and Information Extraction

> **Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
> **Chủ đề:** From Sequence Classification to Token Classification & Information Extraction  
> **Bộ dữ liệu thực nghiệm:** CoNLL-2003 Named Entity Recognition  

---

## 1. Cấu Trúc Thư Mục Nộp Bài (Deliverables)

```text
lab09/
├── README.md                       # Tài liệu tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md                 # Lời giải chi tiết Calculation Exercises 1 (BIO), 2 (Boundary), 3 (Tensor Shapes)
├── prediction.md                   # Các giả thuyết và dự đoán khoa học trước khi thực hiện thực nghiệm
├── baseline_ner.py                 # Mã nguồn Baseline NER (Handcrafted Token Features + Linear Classifier)
├── tokenizer_alignment.ipynb       # Notebook phân tích Subword Tokenization & chiến lược Label Alignment (Strategy A)
├── ner_finetuning.ipynb            # Notebook triển khai và đánh giá Transformer NER (BERT-base-NER) trên CoNLL-2003
├── results.csv                     # Bảng so sánh hiệu năng tổng thể giữa Baseline và Transformer (Precision, Recall, F1)
├── entity_metrics.csv              # Báo cáo chi tiết Precision, Recall, F1 theo từng loại thực thể (PER, ORG, LOC, MISC)
├── entity_length_analysis.png       # Biểu đồ phân tích hiệu năng F1 theo độ dài thực thể (1, 2, 3, 4+ tokens)
├── confusion_matrix.png            # Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) của mô hình Transformer
├── error_analysis.md               # Báo cáo phân tích 15 lỗi thực nghiệm (FP, FN, Boundary) và phân tích ngữ cảnh
├── information_extraction.py       # Pipeline ứng dụng trích xuất thông tin (Unstructured Text -> JSON) và đánh giá định lượng
└── reflection.md                   # Trả lời 10 câu hỏi Learning Check và bài viết Reflection (300-500 từ)
```

---

## 2. Tóm Tắt Kết Quả Thực Nghiệm

### 2.1. So sánh Baseline vs. Pretrained Transformer (`results.csv`)
Đánh giá trên cùng tập kiểm thử test set CoNLL-2003 theo chuẩn **Entity-Level Exact Match** (`seqeval`):

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Linear Classifier + Handcrafted Features)** | 0.6219 | 0.5996 | **0.6105** |
| **Pretrained Transformer (BERT-base-NER)** | **0.9177** | **0.9162** | **0.9170** |

> **Nhận xét:** Pretrained Transformer vượt trội hơn Baseline hơn **+30.6% F1 score**. Nhờ cơ chế Self-Attention và biểu diễn ngữ cảnh động sâu, Transformer giải quyết triệt để vấn đề từ ngoài từ vựng (OOV) và các trường hợp nhập nhằng ngữ cảnh mà các đặc trưng bề mặt thủ công không thể xử lý.

### 2.2. Hiệu năng theo từng loại thực thể (`entity_metrics.csv`)

| Entity Type | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **PER (Person)** | **0.9728** | **0.9568** | **0.9647** | 672 |
| **ORG (Organization)** | 0.9144 | 0.8987 | 0.9065 | 523 |
| **LOC (Location)** | 0.9464 | 0.9103 | 0.9280 | 446 |
| **MISC (Miscellaneous)** | 0.7143 | 0.8376 | 0.7710 | 197 |

> **Phân tích:** 
> - Nhóm `PER` và `LOC` đạt độ chính xác cao nhất ($>92\% - 96\%$) nhờ số lượng mẫu huấn luyện dồi dào và khuôn mẫu từ vựng rõ ràng.
> - Nhóm `MISC` đạt F1 thấp nhất ($77.10\%$) do tính chất không đồng nhất (bao gồm quốc tịch, sự kiện, giải đấu thể thao, giải thưởng).

### 2.3. Phân tích theo độ dài thực thể (`entity_length_analysis.png`)

| Độ dài thực thể (Tokens) | Precision | Recall | F1-Score | Số lượng (TP / Support) |
| :---: | :---: | :---: | :---: | :---: |
| **1 token** | 0.9065 | 0.8927 | **0.8995** | 940 / 1053 |
| **2 tokens** | 0.9446 | 0.9538 | **0.9492** | 682 / 715 |
| **3 tokens** | 0.8281 | 0.8689 | **0.8480** | 53 / 61 |
| **4+ tokens** | 0.7500 | 1.0000 | **0.8571** | 9 / 9 |

> **Nhận xét:** Thực thể $1-2$ tokens chiếm phần lớn dữ liệu và đạt độ chính xác ổn định. Đối với thực thể dài ($3+$ tokens), do yêu cầu Exact Match khắt khe trên toàn bộ span, xác suất mắc lỗi ranh giới tăng lên làm Precision suy giảm.

---

## 3. Hướng Dẫn Chạy & Tái Hiện Kết Quả (Execution Guide)

### Cài đặt môi trường cần thiết
```bash
pip install torch transformers datasets seqeval scikit-learn matplotlib seaborn pandas
```

### 1. Chạy mô hình Baseline NER
Chạy script huấn luyện baseline với đặc trưng thủ công (n-grams, casing, affixes) và Linear Classifier:
```bash
python baseline_ner.py
```

### 2. Chạy ứng dụng Information Extraction (Option A - News)
Trích xuất thông tin có cấu trúc (JSON schema gồm `PERSON`, `ORG`, `LOCATION`) từ văn bản thô và thực hiện đánh giá định lượng trên 30 câu tin tức:
```bash
python information_extraction.py
```

### 3. Xem và chạy các Jupyter Notebooks
- `tokenizer_alignment.ipynb`: Phân tích chi tiết bài toán Subword Tokenization, minh họa cơ chế `word_ids()`, chiến lược gán nhãn Strategy A với `ignore_index=-100`, và thực nghiệm phân tích tỷ lệ $R = \frac{\#\text{subwords}}{\#\text{words}}$ trên 20 câu.
- `ner_finetuning.ipynb`: Triển khai pipeline Transformer Token Classification, kiểm tra tensor shapes ($B \times T \times d$ và $B \times T \times C$), đánh giá entity-level metrics, vẽ biểu đồ phân tích độ dài và hiển thị ma trận nhầm lẫn.

---

## 4. Các Kết Luận Chính

1. **BIO Scheme:** Đóng vai trò thiết yếu để phân định ranh giới (boundary detection) và giải quyết các thực thể liền kề cùng loại.
2. **Strategy A & Ignore Index (`-100`):** Giúp việc căn chỉnh nhãn subword tương thích chuẩn hóa với hàm `CrossEntropyLoss`, bảo toàn thông tin ranh giới ở token đầu và tránh phạt kép mô hình.
3. **Sức mạnh của Contextual Representation:** Cùng một từ (ví dụ: *"Barcelona"*, *"Victoria"*) có thể là `LOC` hoặc `ORG` tùy thuộc vào ngữ cảnh. Cơ chế self-attention của Transformer cho phép phân biệt chính xác tính đa nghĩa này mà biểu diễn tĩnh không làm được.
4. **Cầu nối sang LAB 10:** Kiến trúc Transformer đóng vai trò là backbone tạo contextual representation chung; việc chuyển đổi giữa Sequence Classification (LAB 08), Token Classification (LAB 09) và Extractive QA / Span Prediction (LAB 10) chỉ đơn thuần là việc thay đổi prediction head ở tầng trên cùng.
