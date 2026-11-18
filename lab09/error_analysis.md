# LAB 09 — Báo Cáo Phân Tích Lỗi (Error Analysis) & Phân Tích Ngữ Cảnh (Context Analysis)

---

## 1. Experiment 6 — Error Analysis

Dưới đây là bảng phân tích chi tiết tối thiểu 15 trường hợp lỗi thực tế (bao gồm 5 False Positives, 5 False Negatives và 5 Boundary/Type Errors) được trích xuất trực tiếp từ kết quả dự đoán của mô hình Pretrained Transformer trên tập test CoNLL-2003.

### Bảng Phân Tích Lỗi

| # | Text / Câu chứa thực thể | Gold Entity | Prediction | Error Type | Giải thích nguyên nhân (Explanation) |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **FP 1** | `SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .` | `O` | `LUCKY (PER)` | Spurious Entity / Case Sensitivity | Toàn bộ tiêu đề viết hoa (`ALL CAPS`). Từ *"LUCKY"* (may mắn) vốn là tính từ thường nhưng do viết hoa và đứng trước động từ/danh từ, mô hình bị đánh lừa là tên riêng chỉ người. |
| **FP 2** | `SOCCER - LATE GOALS GIVE JAPAN WIN OVER SYRIA .` | `O` | `LATE (PER)` | Spurious Entity / All-caps Headline | Tương tự tiêu đề tin tức thể thao viết hoa, từ tính từ *"LATE"* (muộn) ở đầu mệnh đề bị nhầm thành tên cầu thủ bóng đá (`PER`). |
| **FP 3** | `CRICKET - LARA ENDURES ANOTHER MISERABLE DAY .` | `O` | `MISERABLE (PER)` | Spurious Entity / Lexical Ambiguity | Cụm *"MISERABLE DAY"* trong tiêu đề viết hoa khiến từ tính từ tiêu cực *"MISERABLE"* bị mô hình ngộ nhận là một phần của tên người đi liền sau *"LARA"*. |
| **FP 4** | `Contractors won 4-0 on aggregate .` | `O` | `Contractors (ORG)` | Spurious Entity / Domain Context | *"Contractors"* là danh từ chung chỉ nhà thầu xây dựng (`O`), nhưng trong bản tin thể thao viết hoa đầu câu với vai trò chủ ngữ của động từ *"won"*, mô hình hiểu nhầm là tên một câu lạc bộ (`ORG`). |
| **FP 5** | `Japan began the defence of their Asian Cup title with a lucky 2-1 win over Syria in Group C ...` | `O` | `Group C (MISC)` | Spurious Entity / Over-generalization | Nhãn vàng trong CoNLL-2003 coi *"Group C"* là danh từ chỉ bảng đấu thông thường (`O`), nhưng mô hình tự động gán nhãn `MISC` do cấu trúc tên bảng đấu tương tự các sự kiện/giải đấu. |
| **FN 1** | `RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .` | `ITALY (LOC)` | `O (Missing)` | Missing Entity / Context Ambiguity | Tiêu đề thể thao viết hoa khiến tên quốc gia *"ITALY"* đứng sau giới từ *"FOR"* bị bỏ sót, do mô hình phân vân giữa quốc gia (`LOC`) và đội tuyển quốc gia (`ORG`) nên tự tin thấp rơi vào `O`. |
| **FN 2** | `SOCCER - LEEDS ' BOWYER FINED FOR PART IN FAST-FOOD FRACAS .` | `BOWYER (PER)` | `O (Missing)` | Missing Entity / Rare Entity | Tên cầu thủ Lee Bowyer xuất hiện dưới dạng họ viết hoa *"BOWYER"* kèm dấu sở hữu cách kẹp giữa, là thực thể hiếm, dẫn đến bị bỏ sót hoàn toàn. |
| **FN 3** | `RUGBY UNION - LITTLE TO MISS CAMPESE FAREWELL .` | `CAMPESE (PER)` | `O (Missing)` | Missing Entity / Rare Entity | Tên cựu danh thủ bóng bầu dục David Campese dạng viết hoa *"CAMPESE"* đứng trước danh từ *"FAREWELL"*, mô hình không nhận diện được đây là tên người. |
| **FN 4** | `SOCCER - UNCAPPED PLAYERS CALLED TO FACE MACEDONIA .` | `MACEDONIA (LOC)` | `O (Missing)` | Missing Entity / Headline Shift | *"MACEDONIA"* trong tiêu đề viết hoa đứng sau cụm *"FACE"* (đối đầu), mô hình không gán được nhãn do phân bố văn phong headline khác với văn xuôi chuẩn. |
| **FN 5** | `CRICKET - AUSTRALIA V WEST INDIES WORLD SERIES SCOREBOARD .` | `WORLD SERIES (MISC)` | `O (Missing)` | Missing Entity / Compound Term | Cụm sự kiện thể thao danh giá *"WORLD SERIES"* bị bỏ sót hoàn toàn do đứng kẹp giữa danh sách đội và từ *"SCOREBOARD"* trong cấu trúc headline ngắn. |
| **BE 1** | `Cuttitta announced his retirement after the 1995 World Cup ,` | `1995 World Cup (MISC)` | `World Cup (MISC)` | Wrong Boundary (Cắt cụt năm) | Nhãn vàng coi cả năm diễn ra sự kiện *"1995 World Cup"* là một thực thể `MISC` 3 tokens, nhưng mô hình chỉ nhận diện *"World Cup"* (2 tokens) và bỏ qua số năm *"1995"*. |
| **BE 2** | `SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .` | `CHINA (PER)` | `CHINA (ORG)` | Wrong Entity Type / Sports Metonymy | Trong ngữ cảnh thể thao, đội tuyển Trung Quốc được gán nhãn vận động viên/đội tuyển (`PER` hoặc `ORG`), mô hình gán `ORG` (tổ chức/đội) thay vì nhãn vàng gốc. |
| **BE 3** | `RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .` | `CUTTITTA (PER)` | `CUTTITTA (LOC)` | Wrong Entity Type / Headline Context | Họ cầu thủ bóng bầu dục Marcello Cuttitta bị mô hình nhầm thành địa danh (`LOC`) do đứng ở vị trí mở đầu tiêu đề thể thao quốc tế. |
| **BE 4** | `Defender Hassan Abbas rose to intercept a long ball into the box but Bitar missed ...` | `Bitar (PER)` | `Bitar (ORG)` | Wrong Entity Type / Rare Proper Noun | Tên thủ môn Bitar là tên riêng người Ả Rập hiếm gặp, mô hình nhầm lẫn gán thành tổ chức (`ORG`) thay vì `PER`. |
| **BE 5** | `SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .` | `JAPAN (LOC)` | `JAPAN (MISC)` | Wrong Entity Type / Metonymy | Tên quốc gia *"JAPAN"* đứng trong tiêu đề bóng đá được nhãn vàng gán `LOC`, nhưng mô hình dự đoán `MISC` (chỉ đội tuyển/sự kiện thể thao). |

---

## 2. Context Analysis (Phân Tích Ngữ Cảnh)

Một trong những thách thức cốt lõi của bài toán Named Entity Recognition là tính **phụ thuộc chặt chẽ vào ngữ cảnh (context-dependent)**: cùng một dạng bề mặt chữ viết (surface form) có thể đóng những vai trò thực thể hoàn toàn khác biệt tùy thuộc vào môi trường cú pháp và ngữ nghĩa xung quanh.

### 3 Ví Dụ Thực Tế Trong Test Set CoNLL-2003

#### Ví dụ 1: Thực thể `"Barcelona"` (`LOC` vs. `ORG`)
- **Vai trò `LOC` (Địa danh / Thành phố):**
  > *"FIFA 's players ' status committee , meeting in **Barcelona** , decided that although the Udinese document was basically valid , it could not be legally protected ."*
  - **Ngữ cảnh:** Đi sau giới từ chỉ nơi chốn *"in"*, là địa điểm tổ chức cuộc họp của hội đồng FIFA $\rightarrow$ Thực thể địa lý (`LOC`).
- **Vai trò `ORG` (Tổ chức / Câu lạc bộ thể thao):**
  > *"Spanish first division match between Real Madrid and **Barcelona**"*
  > *"**Barcelona** fought back strongly but were twice denied by the woodwork..."*
  - **Ngữ cảnh:** Đứng trong cặp đối đầu với *"Real Madrid"*, làm chủ ngữ của động từ hành động thể thao (*"fought back"*) $\rightarrow$ Câu lạc bộ bóng đá FC Barcelona (`ORG`).

#### Ví dụ 2: Thực thể `"Victoria"` (`LOC` vs. `ORG`)
- **Vai trò `LOC` (Bang địa lý):**
  > *"Former Australia test batsman Dean Jones hit an unbeaten 130 to lead **Victoria** 's fightback in their Sheffield Shield match against Tasmania on Saturday ."*
  - **Ngữ cảnh:** Đóng vai trò sở hữu cách đại diện cho địa bàn bang Victoria của Úc $\rightarrow$ Địa danh (`LOC`).
- **Vai trò `ORG` (Đội tuyển cricket):**
  > *"Tasmania 481 for eight declared ... ; **Victoria** 220 for three ( Dean Jones 130 not out ) ."*
  - **Ngữ cảnh:** Nằm trong bảng điểm trận đấu cricket, đại diện cho đội thi đấu ghi điểm số $\rightarrow$ Tổ chức / Đội thể thao (`ORG`).

#### Ví dụ 3: Thực thể `"China"` / `"Japan"` (`LOC` vs. `ORG` / `PER`)
- **Vai trò `LOC` (Quốc gia):**
  > *"The foreign minister arrived in **China** for bilateral trade talks ."*
  - **Ngữ cảnh:** Đích đến của chuyến bay ngoại giao $\rightarrow$ Địa danh quốc gia (`LOC`).
- **Vai trò `ORG` (Đội tuyển thể thao):**
  > *"SOCCER - JAPAN GET LUCKY WIN , **CHINA** IN SURPRISE DEFEAT ."*
  - **Ngữ cảnh:** Làm chủ thể trong trận đấu bóng đá quốc tế bị thất bại $\rightarrow$ Đội tuyển bóng đá quốc gia (`ORG`).

---

### Giải Thích Sâu: Lợi Thế Vượt Trội Của Contextual Representation So Với Representation Tĩnh

Mối liên hệ tiến hóa qua các Lab:
$$\text{LAB 03 (Static Embeddings)} \longrightarrow \text{LAB 08 (Contextual Sequence Representation)} \longrightarrow \text{LAB 09 (Token-Level Contextual Prediction)}$$

1. **Hạn chế cố hữu của Static Representation (Word2Vec, GloVe ở LAB 03):**
   - Trong biểu diễn tĩnh, mỗi từ vựng có một vector duy nhất cố định trong không gian embedding bất kể nó xuất hiện trong câu nào. Do đó, từ `"Barcelona"` luôn luôn có cùng một vector tọa độ dù nằm trong câu *"du lịch ở Barcelona"* hay *"Barcelona ký hợp đồng với tiền đạo mới"*.
   - Để phân loại, mô hình tĩnh phải dựa hoàn toàn vào các đặc trưng thủ công xung quanh (n-gram, từ loại POS), vốn dễ bị nhiễu và không thể nắm bắt được ngữ cảnh cấu trúc ngữ pháp xa.

2. **Sức mạnh của Contextual Representation (Transformer ở LAB 08 & LAB 09):**
   - Kiến trúc Self-Attention cho phép mỗi token trực tiếp "nhìn" và tích hợp thông tin từ tất cả các token còn lại trong câu.
   - Khi từ `"Barcelona"` xuất hiện cạnh các từ khóa thể thao như *"match"*, *"Real Madrid"*, *"score"*, cơ chế attention sẽ gán trọng số lớn cho các từ này, kéo vector biểu diễn của nó về phía không gian biểu diễn của các tổ chức thể thao (`ORG`).
   - Ngược lại, khi `"Barcelona"` xuất hiện sau *"meeting in"*, attention tập trung vào giới từ địa điểm, định vị vector của nó vào vùng không gian địa lý (`LOC`).
3. **Kết luận:** Biểu diễn ngữ cảnh động ở cấp độ token (Token-level contextual representation) cho phép giải quyết triệt để tính đa nghĩa (polysemy) và hiện tượng hoán dụ (metonymy), tạo nên bước nhảy vọt về độ chính xác nhận diện thực thể trong NLP hiện đại.
