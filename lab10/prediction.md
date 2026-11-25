# LAB 10 — Predictions Before Experiment

## Prediction 1: Lexical Overlap

> Nếu question và context chứa nhiều từ giống nhau nhưng chỉ một vị trí thực sự trả lời câu hỏi, model có dễ bị nhầm không?

**Dự đoán:** Có, model có thể bị nhầm. Khi context chứa nhiều cụm từ trùng lặp với question, model — đặc biệt nếu chưa được fine-tune đủ — có xu hướng chọn span ở vùng có lexical overlap cao nhất thay vì vùng thực sự chứa câu trả lời. Tuy nhiên, pretrained Transformer có khả năng học được semantic meaning nên mức độ nhầm lẫn sẽ thấp hơn so với baseline dựa trên keyword overlap thuần túy.

---

## Prediction 2: Answer Length

> Answer span càng dài thì boundary prediction có khó hơn không?

**Dự đoán:** Có, answer span dài sẽ khiến boundary prediction khó hơn. Lý do:
- Với answer ngắn (1–2 tokens), start và end gần nhau hoặc trùng nhau → ít khả năng sai.
- Với answer dài (7+ tokens), model phải xác định chính xác cả hai đầu, xác suất sai ít nhất một đầu tăng lên.
- Dự đoán: EM sẽ giảm rõ rệt khi answer length tăng, còn F1 sẽ giảm chậm hơn vì partial overlap vẫn được tính điểm.

---

## Prediction 3: Distractor Context

> Nếu context chứa nhiều thông tin không liên quan, performance sẽ thay đổi như thế nào?

**Dự đoán:** Performance sẽ giảm. Context dài với nhiều distractor information sẽ:
- Tăng số lượng candidate spans mà model phải xem xét.
- Tạo ra các vùng "nhiễu" có thể thu hút attention của model.
- Đặc biệt ảnh hưởng khi distractor chứa các từ liên quan đến question nhưng không phải answer.
- Dự đoán: F1 giảm khoảng 5–15% so với context sạch chỉ chứa thông tin relevant.

---

## Prediction 4: Context Reordering

> Nếu chỉ thay đổi thứ tự một số câu trong context nhưng giữ nguyên answer, model có bị ảnh hưởng không?

**Dự đoán:** Model sẽ bị ảnh hưởng nhẹ nhưng không nghiêm trọng. Pretrained Transformer sử dụng positional encoding nên thay đổi thứ tự câu sẽ thay đổi representation. Tuy nhiên, vì self-attention có thể attend đến bất kỳ vị trí nào, model vẫn có thể tìm được answer span nếu nội dung không thay đổi. Dự đoán: EM/F1 giảm nhẹ (khoảng 2–5%) vì model có thể đã học được một số positional bias từ training data.
