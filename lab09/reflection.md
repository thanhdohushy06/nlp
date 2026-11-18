# LAB 09 — Reflection and Individual Learning Check

## Phần 1: Trả Lời 10 Câu Hỏi Individual Learning Check (Mục 31)

### 1. Sequence classification khác token classification như thế nào?
- **Sequence Classification:** Ánh xạ toàn bộ chuỗi đầu vào $X = (x_1, x_2, \dots, x_T)$ thành một nhãn phân loại duy nhất $y$ đại diện cho cả câu/đoạn (ví dụ: phân loại cảm xúc Sentiment Analysis $\rightarrow$ `POSITIVE`/`NEGATIVE`).
- **Token Classification:** Ánh xạ chuỗi đầu vào $X = (x_1, \dots, x_T)$ thành một chuỗi nhãn có cùng độ dài $Y = (y_1, \dots, y_T)$, trong đó mỗi token $x_i$ được gán một nhãn riêng biệt $y_i$ (ví dụ: gán nhãn thực thể NER $\rightarrow$ `B-PER`, `O`, `B-LOC`).

### 2. BIO tagging dùng để làm gì?
- BIO tagging là lược đồ gán nhãn theo token (token labeling scheme) dùng để biểu diễn đồng thời cả **loại thực thể (entity category)** và **ranh giới thực thể (entity boundaries)** trong văn bản. Nó giúp mô hình giải quyết việc nhận diện các thực thể gồm nhiều từ (multi-word entities) và phân biệt hai thực thể cùng loại đứng liền kề nhau.

### 3. B-PER khác I-PER như thế nào?
- `B-PER` (Beginning of Person): Đánh dấu token bắt đầu một thực thể chỉ người mới.
- `I-PER` (Inside of Person): Đánh dấu token tiếp theo thuộc về cùng một thực thể người đã được mở đầu trước đó.
- Sự phân biệt này đặc biệt quan trọng khi có hai người xuất hiện liền nhau (ví dụ: *"John Mary visited..."* $\rightarrow$ `John (B-PER)`, `Mary (B-PER)` thay vì bị gộp thành một người nếu dùng `I-PER`).

### 4. Tại sao tokenizer gây ra label alignment problem?
- Dữ liệu NER gốc (như CoNLL-2003) được gán nhãn ở cấp độ **từ nguyên bản (word-level)**. Tuy nhiên, các mô hình Transformer hiện đại sử dụng **subword tokenization** (Byte-Pair Encoding, WordPiece), khiến một từ có thể bị tách thành nhiều mẩu subwords (ví dụ: từ *"playing"* bị tách thành `["play", "##ing"]`). Do đó xuất hiện mâu thuẫn căn chỉnh: số lượng token IDs mà Transformer xử lý nhiều hơn số nhãn ban đầu của các từ. Cần có chiến lược căn chỉnh (alignment strategy) để ánh xạ nhãn word sang nhãn token và ngược lại.

### 5. IGNORE trong loss có ý nghĩa gì?
- `IGNORE` (thường quy ước là `-100` trong PyTorch `CrossEntropyLoss`) là chỉ số đặc biệt báo hiệu cho hàm mất mát bỏ qua hoàn toàn các token tương ứng (như padding tokens, special tokens `[CLS]`, `[SEP]`, và các subwords phía sau trong Strategy A).
- Token mang nhãn `IGNORE` vẫn tham gia vào quá trình tính toán self-attention của Transformer để xây dựng biểu diễn ngữ cảnh, nhưng loss và gradient của nhãn dự đoán tại vị trí đó không được tính vào tổng loss, tránh làm lệch phân phối học của mô hình.

### 6. Tại sao NER không nên chỉ đánh giá bằng token accuracy?
- Trong các tập dữ liệu NER thực tế, phần lớn các token là từ thường (nhãn `O`), chiếm tới $85\% - 90\%$ tổng số token.
- Nếu một mô hình dự đoán tầm thường gán tất cả token thành `O`, nó vẫn đạt token accuracy cực kỳ cao ($85\% - 90\%$) dù không nhận diện được bất kỳ thực thể nào. Ngoài ra, token accuracy không phản ánh được tính toàn vẹn của ranh giới thực thể (đoán đúng 1/2 từ của tên người vẫn được $50\%$ accuracy nhưng thực thể đó hoàn toàn sai).

### 7. Entity-level F1 khác token-level F1 như thế nào?
- **Token-level F1:** Đánh giá độc lập từng token riêng lẻ, coi mỗi token là một mẫu dữ liệu phân loại.
- **Entity-level F1 (chuẩn CoNLL / seqeval):** Đánh giá trên từng span thực thể hoàn chỉnh $(type, start\_index, end\_index)$. Một thực thể chỉ được tính là True Positive khi mô hình dự đoán chính xác tuyệt đối cả loại nhãn lẫn ranh giới bắt đầu và kết thúc. Nếu đoán thiếu hoặc thừa 1 token trong cụm, cả thực thể đó bị tính là lỗi (1 FN và 1 FP).

### 8. Vì sao entity boundary khó?
- Ranh giới thực thể phức tạp do:
  1. Cấu trúc nội bộ của thực thể dài chứa nhiều từ loại thông thường (giới từ *"of"*, liên từ *"and"*, mạo từ *"the"*).
  2. Sự phân mảnh subword tại các tên riêng hiếm gặp.
  3. Tính nhập nhằng của ngôn ngữ: ranh giới ngữ nghĩa có thể mở rộng (ví dụ: *"University of California"* vs. *"University of California Berkeley"*).

### 9. Vì sao contextual representation hữu ích cho NER?
- Cùng một dạng bề mặt từ (surface form) có thể mang các vai trò thực thể hoàn toàn khác nhau tùy thuộc vào ngữ cảnh:
  - *"Washington"* có thể là người (*George Washington*), địa điểm (*Washington D.C.*), hoặc tổ chức/chính phủ (*Washington announced today*).
  - Biểu diễn ngữ cảnh động (dynamic contextual representation) từ Transformer mã hóa toàn bộ câu xung quanh thông qua cơ chế self-attention, cung cấp thông tin ngữ nghĩa và cú pháp đầy đủ giúp bộ phân loại phân biệt chính xác vai trò của từ trong từng ngữ cảnh cụ thể.

### 10. NER có thể được sử dụng như thế nào trong Information Extraction?
- NER đóng vai trò là tầng xử lý nền móng đầu tiên (foundation layer) trong Information Extraction:
  - Biến văn bản thô phi cấu trúc (unstructured text) thành các thực thể có định danh và ngữ nghĩa xác định (`PERSON`, `ORG`, `LOCATION`, `DATE`, `MONEY`).
  - Các thực thể này là đầu vào trực tiếp cho các tác vụ IE phức tạp hơn như: Relation Extraction (trích xuất quan hệ giữa các thực thể, ví dụ: `founder_of(Elon Musk, SpaceX)`), Event Extraction (trích xuất sự kiện), và Knowledge Graph Construction (xây dựng đồ thị tri thức cho doanh nghiệp/hệ thống tìm kiếm).

---

## Phần 2: Reflection — Tính Thống Nhất và Khả Năng Thích Ứng Của Pretrained Transformer (Mục 32)

### Đề bài
> *Viết 300–500 từ trả lời: Tại sao cùng một pretrained Transformer có thể được sử dụng cho cả sequence classification và token classification chỉ bằng cách thay đổi phần prediction head?*

### Bài viết Reflection

Một trong những bước tiến mang tính cách mạng nhất của kiến trúc Transformer (điển hình như BERT hay RoBERTa) chính là sự tách biệt rõ ràng giữa hai tầng: **tầng biểu diễn ngữ cảnh nền tảng (Contextual Representation Backbone)** và **tầng dự đoán chuyên biệt theo tác vụ (Task-specific Prediction Head)**. 

Khi một chuỗi văn bản đầu vào gồm $T$ tokens được đưa qua mạng Transformer nhiều lớp, cơ chế Multi-Head Self-Attention cho phép mỗi token tương tác liên tục với mọi token khác trong câu ở tất cả các lớp biểu diễn. Kết quả ở lớp ẩn cuối cùng (final hidden layer) là một ma trận biểu diễn ngữ cảnh động sâu:
$$H \in \mathbb{R}^{B \times T \times d}$$
Trong đó mỗi vector hàng $h_i \in \mathbb{R}^d$ ($i = 1, \dots, T$) không chỉ đại diện cho riêng từ đó, mà đã được "nhúng" toàn bộ thông tin ngữ nghĩa, cấu trúc cú pháp và văn cảnh của cả câu xung quanh nó.

Điểm mấu chốt để áp dụng cho các downstream tasks khác nhau nằm ở cách chúng ta khai thác ma trận $H$ này:

1. **Đối với Sequence Classification (ví dụ: Phân loại cảm xúc ở LAB 08):**
   - Tác vụ yêu cầu tổng hợp ngữ nghĩa của toàn bộ câu thành một nhãn duy nhất. Mô hình sử dụng token đặc biệt `[CLS]` ở vị trí đầu tiên ($h_{\text{[CLS]}} \in \mathbb{R}^d$), vốn được huấn luyện tiền kỳ để đóng vai trò tóm tắt thông tin ngữ cảnh toàn chuỗi.
   - Khi đó, prediction head là một lớp tuyến tính đơn giản: $z = h_{\text{[CLS]}} W_{\text{seq}} + b_{\text{seq}}$, ánh xạ từ $d$ chiều sang không gian nhãn câu (ví dụ: $2$ lớp Positive/Negative).

2. **Đối với Token Classification (ví dụ: Gán nhãn thực thể NER ở LAB 09):**
   - Tác vụ yêu cầu đưa ra quyết định cục bộ cho từng vị trí từ trong câu. Thay vì chỉ lấy $h_{\text{[CLS]}}$, chúng ta giữ lại toàn bộ ma trận $H = [h_1, h_2, \dots, h_T] \in \mathbb{R}^{T \times d}$.
   - Khi đó, prediction head là một lớp phân loại token: $Z = H W_{\text{tok}} + b_{\text{tok}}$, ánh xạ độc lập từng vector $h_i$ sang không gian $C$ nhãn thực thể BIO ($Z \in \mathbb{R}^{T \times C}$).

Nhờ cơ chế này, cùng một backbone Transformer đã được huấn luyện sẵn trên lượng dữ liệu khổng lồ có thể đóng vai trò như một bộ trích xuất đặc trưng vạn năng. Khi chuyển đổi giữa bài toán phân loại toàn câu sang bài toán gán nhãn từng từ, ta không cần thay đổi kiến trúc nội tại hay trọng số của Transformer, mà chỉ cần thay đổi **chiều rút trích của ma trận ẩn $H$** và thay thế một lớp tuyến tính mỏng ở đầu ra tương ứng với bài toán. Đây chính là minh chứng rõ ràng cho tính linh hoạt và sức mạnh của mô hình nền tảng (Foundation Models) trong xử lý ngôn ngữ tự nhiên hiện đại.
