# LAB 10 — Reflection

## LAB 08, LAB 09 và LAB 10: Ba bài toán, một backbone

LAB 08, LAB 09 và LAB 10 đều sử dụng pretrained Transformer làm backbone nhưng giải quyết ba bài toán NLP khác nhau bằng cách thay đổi prediction head và cách định nghĩa target.

### Sơ đồ tổng quan

```
                    Pretrained Transformer
                            │
            ┌───────────────┼────────────────┐
            ↓               ↓                ↓
    [CLS] representation  Token reps      Token reps
            ↓               ↓                ↓
      Classification       NER         Start / End
            ↓               ↓                ↓
         LAB 08          LAB 09           LAB 10
```

### LAB 08 — Sequence Classification

Ở LAB 08, toàn bộ input text được đưa qua pretrained Transformer, và chỉ **representation của token [CLS]** được sử dụng. Token [CLS] đóng vai trò tóm tắt thông tin của cả sequence. Một linear layer ánh xạ vector [CLS] sang distribution trên các classes. Target là **một label duy nhất cho toàn bộ document**. Đây là bài toán đơn giản nhất: mỗi input → một output.

### LAB 09 — Token Classification (NER)

LAB 09 chuyển sang sử dụng **representation của mọi token** thay vì chỉ [CLS]. Mỗi token được gắn một label riêng (B-PER, I-PER, O, ...). Linear layer được áp dụng lên từng token position. Target là **một chuỗi labels có cùng độ dài với input tokens**. Bài toán phức tạp hơn LAB 08 vì model phải quyết định cho từng token thay vì toàn bộ sequence.

### LAB 10 — Span Prediction (Extractive QA)

LAB 10 tiếp tục sử dụng representation của mọi token nhưng theo cách khác: thay vì gán label cho từng token, model dự đoán **hai vị trí**: start position và end position. Input gồm cả question và context (được nối bằng [SEP]). Hai linear classifiers riêng biệt hoạt động trên cùng token representations để dự đoán xác suất mỗi token là start hoặc end của answer span. Target là **một cặp (start, end)** xác định đoạn text trong context chứa câu trả lời.

### Tại sao cùng backbone nhưng prediction head khác nhau?

Pretrained Transformer (như BERT) không phải bản thân một ứng dụng cụ thể — nó là một **feature extractor** tạo contextual representations chất lượng cao cho mọi token. Cách đặt prediction head và target quyết định loại bài toán:

- **Classification**: chỉ cần 1 vector → 1 label
- **NER**: mỗi token → 1 label → cần T classifiers (shared weights)
- **QA**: cần 2 positions → 2 classifiers trên toàn bộ tokens

Điều này cho thấy sức mạnh của transfer learning: cùng một pretrained backbone có thể phục vụ nhiều downstream tasks khác nhau chỉ bằng cách thay đổi "đầu ra" (prediction head) và fine-tune trên task-specific data. Đây cũng là nền tảng để hiểu tại sao ở LAB 11 và LAB 12, cùng các Transformer encoders có thể được dùng cho retrieval (tạo embeddings) và generation (RAG), mở rộng thêm bức tranh toàn diện về ứng dụng của pretrained models trong NLP.
