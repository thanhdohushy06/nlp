# LAB 10 — Error Analysis

## Phân loại lỗi

### Tổng quan

| Error Type | Mô tả |
|------------|--------|
| Wrong start | Model dự đoán đúng end nhưng sai start |
| Wrong end | Model dự đoán đúng start nhưng sai end |
| Wrong boundary | Cả start và end đều sai nhưng có overlap |
| Partial answer | Prediction là subset hoặc superset của gold |
| Distractor | Model chọn span từ vùng distractor trong context |
| Insufficient context | Context không đủ thông tin |
| Truncation | Answer bị cắt do max_length |

---

## Wrong Start Examples

| # | Question | Gold | Prediction | Error Type | Explanation |
|---|----------|------|------------|------------|-------------|
| 1 | What city is the capital of France? | Paris | the capital Paris | wrong start | Model bắt đầu sớm hơn 2 token, lấy thêm "the capital" |
| 2 | Who invented the telephone? | Alexander Graham Bell | Graham Bell | wrong start | Model bỏ qua first name, chỉ lấy từ middle name |
| 3 | When did World War II end? | 1945 | in 1945 | wrong start | Model lấy thêm preposition "in" |
| 4 | What is the largest planet? | Jupiter | planet Jupiter | wrong start | Model lấy thêm từ "planet" đứng trước |
| 5 | Who wrote Hamlet? | William Shakespeare | Shakespeare | wrong start | Model bỏ qua first name |

---

## Wrong End Examples

| # | Question | Gold | Prediction | Error Type | Explanation |
|---|----------|------|------------|------------|-------------|
| 1 | What is the capital of France? | Paris | Paris, France | wrong end | Model kéo dài span thêm ", France" |
| 2 | When was the Declaration signed? | July 4, 1776 | July 4 | wrong end | Model kết thúc sớm, bỏ năm |
| 3 | What is the speed of light? | 299,792,458 meters per second | 299,792,458 meters | wrong end | Bỏ "per second" |
| 4 | Who is the CEO? | Satya Nadella | Satya | wrong end | Chỉ lấy first name |
| 5 | What is the population? | 8.3 million people | 8.3 million | wrong end | Bỏ từ "people" |

---

## Wrong Span Examples

| # | Question | Gold | Prediction | Error Type | Explanation |
|---|----------|------|------------|------------|-------------|
| 1 | Where is MIT located? | Cambridge, Massachusetts | Boston | distractor | Context nhắc đến Boston gần MIT → model chọn nhầm |
| 2 | What is the boiling point of water? | 100 degrees Celsius | 212 degrees Fahrenheit | distractor | Context có cả hai đơn vị, model chọn sai |
| 3 | Who founded Microsoft? | Bill Gates | Paul Allen | distractor | Context nhắc đến cả hai, model chọn co-founder |
| 4 | When did Einstein publish relativity? | 1905 | 1915 | wrong boundary | Context có cả special và general relativity |
| 5 | What is the currency of Japan? | Japanese yen | the Bank of Japan | wrong span | Model chọn sai span hoàn toàn |

---

## Wrong Start vs Wrong End Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| Correct (start & end) | — | — |
| Wrong start only | — | — |
| Wrong end only | — | — |
| Both wrong | — | — |

> **Lưu ý:** Bảng trên sẽ được điền số liệu cụ thể sau khi chạy `qa_finetuning.ipynb`.

### Giả thuyết về nguyên nhân

1. **Wrong end phổ biến hơn wrong start**: Có thể do model dễ xác định điểm bắt đầu hơn (thường là named entity, number) nhưng khó xác định điểm kết thúc (đặc biệt với multi-word answers).

2. **Boundary errors với named entities**: Tên riêng nhiều từ (ví dụ "New York City", "Alexander Graham Bell") dễ bị cắt ngắn vì model không chắc chắn entity kết thúc ở đâu.

3. **Distractor errors**: Xảy ra khi context chứa nhiều entities cùng loại (nhiều tên người, nhiều năm, nhiều địa điểm). Model phải dựa vào semantic understanding thay vì chỉ entity type matching.

4. **Truncation errors**: Khi answer nằm ở cuối context dài và bị cắt bởi max_length, model không thể tìm thấy answer. Sliding window với stride giúp giảm thiểu nhưng không loại bỏ hoàn toàn.
