# LAB 10 — Extractive Question Answering

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026  
**Chủ đề:** From Token Classification to Span Prediction

---

## Mô tả

LAB 10 xây dựng hệ thống Extractive Question Answering sử dụng pretrained Transformer (BERT). Cho cặp (question, context), model dự đoán start/end position của answer span trong context.

## Cấu trúc

```
lab10/
├── README.md                    # File này
├── calculations.md              # Calculation exercises (start/end, loss, character span)
├── prediction.md                # Predictions trước khi thí nghiệm
├── tokenizer_alignment.ipynb    # Tokenization, offset mapping, span alignment
├── qa_model.py                  # Core implementation (model, dataset, metrics, baseline)
├── qa_finetuning.ipynb          # Fine-tuning, experiments, evaluation
├── results.csv                  # Kết quả thí nghiệm
├── learning_curves.png          # Training/validation loss curves (generated)
├── em_f1_analysis.png           # EM vs F1 comparison (generated)
├── answer_length_analysis.png   # Performance theo answer length (generated)
├── error_analysis.md            # Phân tích lỗi chi tiết
├── document_qa.py               # Mini Document QA application
└── reflection.md                # Reflection (300-500 từ)
```

## Cách chạy

### 1. Cài đặt dependencies

```bash
pip install torch transformers datasets pandas matplotlib
```

### 2. Chạy tokenizer alignment notebook

```bash
jupyter notebook tokenizer_alignment.ipynb
```

### 3. Fine-tune QA model

```bash
jupyter notebook qa_finetuning.ipynb
```

### 4. Chạy Document QA demo

```bash
python document_qa.py
```

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Model | bert-base-uncased |
| Tokenizer | bert-base-uncased |
| Dataset | SQuAD v1.1 |
| Train split | 5000 samples |
| Validation split | 1000 samples |
| Max sequence length | 384 |
| Stride | 128 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Epochs | 3 |
| Optimizer | AdamW (weight_decay=0.01) |
| Random seed | 42 |

## Individual Learning Check

1. **Extractive QA khác classification như thế nào?**  
   Classification gán 1 label cho toàn bộ text; Extractive QA dự đoán 2 positions (start, end) trong context để trích xuất answer span.

2. **Tại sao QA cần start và end position?**  
   Vì answer có thể có độ dài bất kỳ. Cần cả start và end để xác định chính xác span text nào là câu trả lời.

3. **Character offset khác token index như thế nào?**  
   Character offset đếm vị trí ký tự trong chuỗi gốc; token index đếm vị trí token sau khi tokenize (bao gồm subwords).

4. **Tại sao cần offset mapping?**  
   Để chuyển đổi giữa character positions (từ dataset) và token positions (input của model).

5. **Stride dùng để làm gì?**  
   Tạo overlap giữa các sliding windows khi context quá dài, đảm bảo answer không bị mất ở ranh giới windows.

6. **Vì sao context dài có thể phải chia thành nhiều window?**  
   Transformer có giới hạn max sequence length (thường 512). Context dài hơn phải được chia nhỏ.

7. **Exact Match đo điều gì?**  
   Đo tỷ lệ predictions khớp hoàn toàn với gold answer (sau normalization).

8. **F1 của QA khác gì với accuracy?**  
   F1 đo token-level overlap (precision × recall), cho phép partial credit. Accuracy chỉ đo đúng/sai hoàn toàn.

9. **Vì sao prediction có thể có F1 > 0 nhưng EM = 0?**  
   Khi prediction có overlap một phần với gold (ví dụ: gold="New York City", pred="New York"), F1 > 0 vì có shared tokens, nhưng EM = 0 vì không khớp hoàn toàn.

10. **Wrong-start và wrong-end khác nhau như thế nào?**  
    Wrong-start: model bắt đầu span sai vị trí (ví dụ bỏ first name). Wrong-end: model kết thúc span sai (ví dụ thiếu hoặc thừa tokens).

11. **Extractive QA khác RAG ở đâu?**  
    Extractive QA giả định context đã được cung cấp. RAG thêm bước retrieval để tìm context phù hợp từ corpus lớn trước khi trả lời.

12. **Khi nào extractive QA không thể trả lời câu hỏi?**  
    Khi câu trả lời không nằm trong context (cần suy luận, tổng hợp, hoặc context không chứa thông tin liên quan).
