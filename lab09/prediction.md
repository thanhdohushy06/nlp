# LAB 09 — Dự Đoán Trước Thực Nghiệm (Prediction Before Experiment)

Tài liệu này ghi lại các giả thuyết và dự đoán khoa học trước khi tiến hành huấn luyện và đánh giá thực nghiệm các mô hình Named Entity Recognition (NER).

---

## Prediction 1: Pretrained Transformer vs. Simple Classifier (Word Features)

**Câu hỏi:**
> *Nếu sử dụng pretrained Transformer thay vì một classifier đơn giản dựa trên word features (như TF-IDF / Token features + Logistic Regression), NER sẽ thay đổi như thế nào?*

**Dự đoán:**
Hiệu năng của mô hình Pretrained Transformer sẽ vượt trội hoàn toàn so với mô hình Baseline (Simple Classifier dựa trên word features) trên tất cả các thước đo (Precision, Recall, và đặc biệt là Macro/Micro F1 score). Cụ thể:
1. **Khả năng mô hình hóa ngữ cảnh sâu (Deep Contextualization):**
   - Bộ phân loại đơn giản chỉ dựa vào các đặc trưng cục bộ (từ hiện tại, tiền tố/hậu tố, từ lân cận trong cửa sổ cố định $\pm 1$ hoặc $\pm 2$). Do đó, mô hình này dễ bị nhầm lẫn khi một từ có tính đa nghĩa (polysemy) hoặc mang các vai trò thực thể khác nhau trong các ngữ cảnh khác nhau (ví dụ: *"Apple"* là trái cây hay tập đoàn công nghệ; *"Washington"* là người, thành phố, bang, hay chính phủ).
   - Pretrained Transformer (như BERT) sử dụng cơ chế Multi-Head Self-Attention để xây dựng biểu diễn ngữ cảnh động cho từng token qua toàn bộ câu, nắm bắt được các phụ thuộc cú pháp và ngữ nghĩa tầm xa.
2. **Khả năng xử lý từ ngoài tập từ vựng (Out-Of-Vocabulary - OOV):**
   - Word-level baseline gặp khó khăn nghiêm trọng khi gặp các tên riêng mới (unseen entities) chưa từng thấy trong tập train, thường gán nhầm thành nhãn phổ biến `O`.
   - Transformer giải quyết OOV nhờ subword tokenization (WordPiece/BPE) kết hợp cùng biểu diễn ngữ cảnh, giúp suy luận nhãn thực thể dựa trên môi trường ngữ cảnh xung quanh (context clues) dù bản thân từ đó hiếm hoặc chưa từng xuất hiện.
3. **Độ recall tăng vọt:**
   - Recall của Baseline thường rất thấp vì nó bỏ sót phần lớn các biến thể tên riêng không nằm trong từ điển đặc trưng. Transformer sẽ cải thiện vượt bậc Recall nhờ khả năng khái quát hóa (generalization).

---

## Prediction 2: Độ dài thực thể (Entity Length) — Thực thể dài vs. Thực thể ngắn

**Câu hỏi:**
> *Các entity dài nhiều token sẽ dễ hay khó hơn entity chỉ có một token? (Ví dụ: `Microsoft` so với `United Nations Security Council`). Giải thích.*

**Dự đoán:**
Các entity dài gồm nhiều token sẽ **khó nhận diện chính xác hơn đáng kể** so với các entity ngắn chỉ có một token. Cụ thể, $F1$ score của các thực thể dài ($3+$ hoặc $4+$ tokens) sẽ thấp hơn nhiều so với các thực thể $1-2$ tokens.

**Giải thích lý do:**
1. **Quy tắc đánh giá Entity-Level khắt khe (Exact Match Requirement):**
   - Theo chuẩn đánh giá NER (CoNLL / seqeval), một thực thể chỉ được tính là True Positive (TP) khi và chỉ khi **cả ranh giới (boundary) lẫn loại nhãn (type)** đều chính xác hoàn toàn từ token đầu tiên đến token cuối cùng.
   - Với thực thể 1 token như *"Microsoft"*, mô hình chỉ cần dự đoán đúng $1$ quyết định duy nhất: `B-ORG`.
   - Với thực thể 4 tokens như *"United Nations Security Council"*, mô hình phải đưa ra $4$ quyết định liên tiếp chính xác tuyệt đối: `[B-ORG, I-ORG, I-ORG, I-ORG]`.
2. **Xác suất cộng dồn sai số (Compounding Error):**
   - Nếu xác suất mô hình gán đúng nhãn cho mỗi token độc lập là $p \approx 0.95$, thì xác suất cả cụm 4 token đều đúng chỉ là $p^4 \approx (0.95)^4 \approx 0.814$.
   - Chỉ cần $1$ token trong cụm bị gán sai (ví dụ: *"Security"* bị gán là `O`, hoặc *"Council"* bị gán nhầm thành `B-ORG`), toàn bộ thực thể sẽ bị coi là lỗi (vừa mất $1$ FN do không nhận diện được thực thể chuẩn, vừa sinh ra $1$ FP do dự đoán thực thể sai ranh giới).
3. **Cấu trúc ngữ pháp phức tạp bên trong thực thể dài:**
   - Các thực thể dài thường chứa các từ thông dụng, giới từ, danh từ chung (như *"of"*, *"and"*, *"Association"*, *"Security"*, *"Bank"*). Các từ này xuất hiện với tần suất cao trong vai trò từ thường (`O`) ở các văn cảnh khác, khiến mô hình dễ cắt cụt thực thể tại các từ này (Boundary Truncation).

---

## Prediction 3: Tần suất xuất hiện (Entity Frequency) — Thực thể hiếm vs. Thực thể phổ biến

**Câu hỏi:**
> *Entity hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không? Phải đưa ra lý do trước khi chạy experiment.*

**Dự đoán:**
Có, các entity hiếm xuất hiện (rare entities / low-frequency entities) hoặc không xuất hiện (unseen entities) trong training data **chắc chắn có tỷ lệ gây lỗi cao hơn nhiều** so với các entity xuất hiện nhiều lần (frequent entities).

**Lý do:**
1. **Thiếu mẫu học bề mặt từ vựng (Lack of Lexical Memorization):**
   - Các mô hình học máy (bao gồm cả neural network) có xu hướng ghi nhớ (memorize) một phần các dạng bề mặt (surface forms) xuất hiện lặp đi lặp lại nhiều lần trong tập train (ví dụ: *"China"*, *"London"*, *"Clinton"*). Với các thực thể quen thuộc này, mô hình có thể nhận diện đúng gần như 100% ngay cả khi ngữ cảnh mập mờ.
   - Đối với các thực thể hiếm (ví dụ tên người hoặc địa danh ít phổ biến ở các nước đang phát triển, tên công ty khởi nghiệp mới), mô hình không thể dựa vào bộ nhớ từ vựng mà buộc phải suy diễn 100% từ ngữ cảnh xung quanh.
2. **Mất cân bằng dữ liệu (Data Imbalance):**
   - Phân bố tần suất từ trong ngôn ngữ tự nhiên tuân theo quy luật Zipf's Law: một số ít thực thể chiếm đa số lượt xuất hiện, trong khi phần lớn thực thể chỉ xuất hiện 1-2 lần.
   - Gradient cập nhật trong quá trình huấn luyện bị chi phối bởi các thực thể xuất hiện thường xuyên, dẫn đến việc mô hình thiên vị các mẫu phổ biến và kém nhạy cảm với các mẫu hiếm.
3. **Phân rã Subword phức tạp hơn:**
   - Các từ hiếm thường là tên riêng ít gặp, dẫn đến việc bị tokenizer tách thành nhiều mẩu subword nhỏ (ví dụ: một tên người lạ có thể bị chia thành 3-4 subwords). Việc phân rã nhiều subwords làm tăng nguy cơ sai sót trong quá trình liên kết nhãn (subword label alignment) và dự đoán chuỗi nhãn liên tục.
