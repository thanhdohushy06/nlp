# LAB 09 — Calculation Exercises

## 1. Calculation Exercise 1 — BIO Tagging

### Bài toán
Cho câu:
> **"Elon Musk founded SpaceX in California."**

Giả sử các thực thể:
- `Elon Musk` $\rightarrow$ `PER` (Person)
- `SpaceX` $\rightarrow$ `ORG` (Organization)
- `California` $\rightarrow$ `LOC` (Location)

### Kết quả gán nhãn BIO tương ứng cho từng token

| Word / Token | Gold Entity | BIO Tag | Giải thích |
| :--- | :--- | :--- | :--- |
| **Elon** | PER | `B-PER` | Bắt đầu thực thể người (Beginning of PER) |
| **Musk** | PER | `I-PER` | Tiếp tục/nằm bên trong thực thể người (Inside PER) |
| **founded** | - | `O` | Ngoài thực thể (Outside) |
| **SpaceX** | ORG | `B-ORG` | Bắt đầu thực thể tổ chức (Beginning of ORG, thực thể 1 token) |
| **in** | - | `O` | Ngoài thực thể (Outside) |
| **California** | LOC | `B-LOC` | Bắt đầu thực thể địa điểm (Beginning of LOC, thực thể 1 token) |
| **.** | - | `O` | Dấu câu, ngoài thực thể (Outside) |

---

## 2. Calculation Exercise 2 — Entity Boundary

### Bài toán
Cho cụm từ:
> **"University of California Berkeley"**

Giả sử toàn bộ cụm là một organization (`ORG`).

### Kết quả gán nhãn BIO:
- `University` $\rightarrow$ `B-ORG`
- `of` $\rightarrow$ `I-ORG`
- `California` $\rightarrow$ `I-ORG`
- `Berkeley` $\rightarrow$ `I-ORG`

### Câu hỏi: Vì sao không thể gán tất cả token là `ORG` nếu muốn biểu diễn ranh giới entity?

**Trả lời:**
1. **Vấn đề phân tách ranh giới giữa các thực thể liền kề (Boundary Ambiguity):**
   Nếu chỉ gán nhãn lớp thực thể thuần túy (chỉ dùng `ORG`), khi có hai thực thể cùng loại xuất hiện kề sát nhau mà không có từ nối (ví dụ: *"Stanford UC Berkeley announced a joint research program"* hoặc *"Bank of America Merrill Lynch merger"*), chuỗi nhãn sẽ là `ORG ORG ORG ...`. Khi đó, hệ thống không thể biết đây là một thực thể duy nhất tên là *"Stanford UC Berkeley"* hay là hai thực thể tách biệt (*"Stanford"* và *"UC Berkeley"*).

2. **Vai trò cốt lõi của tiền tố `B-` (Beginning) và `I-` (Inside):**
   - `B-` (Beginning): Đánh dấu token mở đầu một thực thể mới.
   - `I-` (Inside): Đánh dấu các token tiếp theo thuộc cùng một thực thể được mở đầu bởi token `B-` trước đó.
   - Sự xuất hiện của nhãn `B-` cho phép hệ thống xác định chính xác điểm bắt đầu (onset) của một span thực thể mới, ngay cả khi nó đứng ngay sát một thực thể cùng loại khác (`B-ORG` theo sau bởi `B-ORG`).

3. **Kết luận:**
   Bài toán Named Entity Recognition (NER) là bài toán kép: vừa phải **phân loại đúng loại thực thể (classification)**, vừa phải **xác định chính xác biên giới thực thể (span boundary detection)**. BIO scheme là cơ chế tối thiểu và hiệu quả nhất để biểu diễn cả hai chiều thông tin này dưới dạng chuỗi gán nhãn theo token (token sequence labeling).

---

## 3. Calculation Exercise 3 — Tensor Shapes trong Transformer Token Classification

### Thiết lập kích thước
- Kích thước batch: $B = 8$
- Độ dài chuỗi token tối đa: $T = 128$
- Kích thước vector ẩn (hidden dimension): $d = 768$ (ví dụ: BERT-base)
- Số lượng nhãn NER: $C = 9$ (tương ứng với 9 nhãn: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`)

### Kiến trúc phân loại (Classification Head)
Pretrained Transformer sinh ra ma trận biểu diễn ngữ cảnh:
$$H \in \mathbb{R}^{B \times T \times d}$$

Lớp Linear Head chiếu từng vector token $d = 768$ sang không gian $C = 9$ nhãn:
$$Z = HW + b$$
Trong đó:
- $W \in \mathbb{R}^{d \times C} = \mathbb{R}^{768 \times 9}$
- $b \in \mathbb{R}^C = \mathbb{R}^9$

### Suy ra Shape của các tensor:

1. **Hidden states ($H$):**
   $$\text{Shape}(H) = B \times T \times d = 8 \times 128 \times 768$$
   - Mỗi token trong mỗi câu của batch được biểu diễn bằng một vector ngữ cảnh 768 chiều.

2. **Logits ($Z$):**
   $$\text{Shape}(Z) = B \times T \times C = 8 \times 128 \times 9$$
   - Với mỗi token trong câu, model xuất ra một phân phối unnormalized log-probabilities trên 9 nhãn NER.

3. **Labels ($Y$):**
   $$\text{Shape}(Y) = B \times 128 = 8 \times 128$$
   - Mỗi phần tử $Y_{b, t} \in \{0, 1, \dots, 8\}$ tương ứng với ground-truth label ID của token thứ $t$ trong câu $b$ (hoặc mang giá trị `-100` / `IGNORE` đối với padding tokens và subword tokens không tham gia tính loss).
