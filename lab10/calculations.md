# LAB 10 — Calculations

## Calculation Exercise 1 — Start / End

### Example 1: Answer = "Paris"

Context sau khi tokenize:

```
Index:  0     1        2    3       4   5      6
Token: [The, capital, of, France, is, Paris, .]
```

Question: *What is the capital of France?*
Answer: **Paris**

- `start = 5` (token "Paris" ở vị trí index 5)
- `end = 5` (answer chỉ gồm 1 token nên end = start)

### Example 2: Answer = "capital of France"

Context sau khi tokenize:

```
Index:  0     1        2    3       4   5      6
Token: [The, capital, of, France, is, Paris, .]
```

Answer: **capital of France**

- `start = 1` (token "capital" ở vị trí index 1)
- `end = 3` (token "France" ở vị trí index 3)

---

## Calculation Exercise 2 — Start Loss

Answer start position là token 5. Model dự đoán:

| Position | Probability |
|----------|-------------|
| 0        | 0.02        |
| 1        | 0.05        |
| 2        | 0.03        |
| 3        | 0.10        |
| 4        | 0.15        |
| 5        | 0.60        |
| 6        | 0.05        |

$$L_{start} = -\log(0.60) = -\log(0.60) \approx 0.5108$$

### Giải thích: Tại sao cần hai prediction (start và end) thay vì chỉ một?

Vì answer span có thể có độ dài bất kỳ — từ 1 token đến hàng chục token. Nếu chỉ dùng một prediction (ví dụ chỉ dự đoán start), model không biết answer kết thúc ở đâu. Nếu dùng classification cho toàn bộ span (ví dụ liệt kê tất cả cặp (i, j) có thể), số lượng classes sẽ là $O(T^2)$ — quá lớn. Bằng cách tách thành hai bài toán độc lập: dự đoán start position và end position, mỗi bài chỉ cần $O(T)$ classes, giúp model hiệu quả hơn rất nhiều.

---

## Calculation Exercise 3 — Character Span

Context: `The capital of France is Paris.`

```
T  h  e     c  a  p  i  t  a  l     o  f     F  r  a  n  c  e     i  s     P  a  r  i  s  .
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

Answer: **Paris**

1. Index ký tự đầu tiên của "Paris": **25**
2. Index ký tự cuối của "Paris": **29**
3. Answer span theo character: **(25, 30)** (exclusive end) hoặc **(25, 29)** (inclusive end)
4. Sau tokenization (giả sử word-level): token tương ứng là "Paris"
5. Start token: **5** (token index 5 trong danh sách `[The, capital, of, France, is, Paris, .]`)
6. End token: **5**

**Lưu ý quan trọng:** Character offset (25) ≠ token index (5). Character offset đếm từng ký tự trong chuỗi gốc, còn token index đếm vị trí token sau khi tokenize.

---

## Token-level F1 Calculation Example

Gold: **Paris France** → tokens: {Paris, France}
Prediction: **France** → tokens: {France}

- Overlap = {France} → #overlap = 1
- Precision = #overlap / #predicted = 1/1 = 1.0
- Recall = #overlap / #gold = 1/2 = 0.5
- F1 = 2 × P × R / (P + R) = 2 × 1.0 × 0.5 / (1.0 + 0.5) = **0.6667**

Nhận xét: F1 > 0 nhưng EM = 0 vì prediction không khớp hoàn toàn với gold answer.
