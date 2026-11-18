"""
information_extraction.py — Mini Information Extraction Application using NER
Extracts structured entity records (PERSON, ORG, LOCATION) from raw unstructured text.
Includes quantitative evaluation on 30 annotated news sentences (Option A - News).
"""

import json
import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics.sequence_labeling import get_entities

class NewsInformationExtractor:
    """
    Pipeline trích xuất thông tin có cấu trúc từ văn bản tin tức thô (Unstructured Text -> Structured Data).
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        print(f"Đang khởi tạo Information Extractor với backbone: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Trích xuất các thực thể theo danh mục: PERSON, ORG, LOCATION từ câu văn bản thô.
        """
        import re
        words = re.findall(r"[\w']+|[.,!?;:]", text.strip())
        if not words:
            return {"PERSON": [], "ORG": [], "LOCATION": []}

        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]

        word_ids = inputs.word_ids(batch_index=0)
        
        # Chiến lược Alignment: Lấy nhãn của subword đầu tiên cho mỗi word
        word_labels = []
        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != prev_word_idx:
                label_id = predictions[token_idx].item()
                word_labels.append(self.id2label[label_id])
                prev_word_idx = word_idx

        # Gom nhóm thực thể theo BIO tags
        structured_data: Dict[str, List[str]] = {
            "PERSON": [],
            "ORG": [],
            "LOCATION": []
        }

        # Ánh xạ nhãn CoNLL sang chuẩn schema đầu ra
        label_map = {
            "PER": "PERSON",
            "ORG": "ORG",
            "LOC": "LOCATION"
        }

        current_entity_tokens = []
        current_entity_type = None

        for word, tag in zip(words, word_labels):
            if tag.startswith("B-"):
                if current_entity_type and current_entity_tokens:
                    target_key = label_map.get(current_entity_type)
                    if target_key:
                        structured_data[target_key].append(" ".join(current_entity_tokens))
                current_entity_type = tag[2:]
                current_entity_tokens = [word]
            elif tag.startswith("I-") and current_entity_type == tag[2:]:
                current_entity_tokens.append(word)
            else:
                if current_entity_type and current_entity_tokens:
                    target_key = label_map.get(current_entity_type)
                    if target_key:
                        structured_data[target_key].append(" ".join(current_entity_tokens))
                current_entity_type = None
                current_entity_tokens = []

        if current_entity_type and current_entity_tokens:
            target_key = label_map.get(current_entity_type)
            if target_key:
                structured_data[target_key].append(" ".join(current_entity_tokens))

        # Loại bỏ các thực thể trùng lặp trong cùng một câu nếu có
        for k in structured_data:
            structured_data[k] = list(dict.fromkeys(structured_data[k]))

        return structured_data


def evaluate_ie_pipeline(extractor: NewsInformationExtractor, num_sentences: int = 30):
    """
    Đánh giá định lượng pipeline Information Extraction trên tập kiểm thử CoNLL-2003 (30 câu tin tức).
    Tính toán Precision, Recall, F1 ở cấp độ thực thể (Exact Match Entity-Level).
    """
    print("\n" + "=" * 65)
    print(f"TIẾN HÀNH ĐÁNH GIÁ ĐỊNH LƯỢNG IE PIPELINE TRÊN {num_sentences} CÂU TIN TỨC...")
    print("=" * 65)

    ds = load_dataset('lhoestq/conll2003')
    tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    test_subset = ds['test'][:num_sentences]

    tp_total = 0
    fp_total = 0
    fn_total = 0

    per_type_stats = {
        "PERSON": {"tp": 0, "fp": 0, "fn": 0},
        "ORG": {"tp": 0, "fp": 0, "fn": 0},
        "LOCATION": {"tp": 0, "fp": 0, "fn": 0}
    }

    type_mapping = {
        "PER": "PERSON",
        "ORG": "ORG",
        "LOC": "LOCATION"
    }

    for idx in range(num_sentences):
        tokens = test_subset['tokens'][idx]
        ner_tags = test_subset['ner_tags'][idx]
        gold_bio = [tag_names[t] for t in ner_tags]

        # Trích xuất gold entities dạng (type, text)
        gold_entities = []
        for ent in get_entities(gold_bio):
            etype, start, end = ent
            mapped_type = type_mapping.get(etype)
            if mapped_type:
                ent_text = " ".join(tokens[start:end+1])
                gold_entities.append((mapped_type, ent_text))

        # Dự đoán từ Information Extractor
        raw_text = " ".join(tokens)
        extracted = extractor.extract_entities(raw_text)

        pred_entities = []
        for cat, items in extracted.items():
            for item in items:
                pred_entities.append((cat, item))

        gold_set = set(gold_entities)
        pred_set = set(pred_entities)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        for cat in per_type_stats:
            cat_gold = {e for e in gold_set if e[0] == cat}
            cat_pred = {e for e in pred_set if e[0] == cat}
            c_tp = len(cat_gold & cat_pred)
            c_fp = len(cat_pred - cat_gold)
            c_fn = len(cat_gold - cat_pred)
            per_type_stats[cat]["tp"] += c_tp
            per_type_stats[cat]["fp"] += c_fp
            per_type_stats[cat]["fn"] += c_fn

    # Tính toán chỉ số tổng thể
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n[KẾT QUẢ ĐÁNH GIÁ TOÀN DIỆN TRÊN {num_sentences} CÂU]:")
    print(f"  - True Positives  (TP) : {tp_total}")
    print(f"  - False Positives (FP) : {fp_total}")
    print(f"  - False Negatives (FN) : {fn_total}")
    print(f"  - Overall Precision    : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Overall Recall       : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - Overall F1 Score     : {f1:.4f} ({f1*100:.2f}%)")

    print("\n[CHI TIẾT THEO TỪNG LOẠI THỰC THỂ]:")
    print(f"{'Entity Type':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 65)
    for cat, s in per_type_stats.items():
        c_p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        c_r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
        print(f"{cat:<12} | {c_p:<10.4f} | {c_r:<10.4f} | {c_f1:<10.4f} | {s['tp']:<4} | {s['fp']:<4} | {s['fn']:<4}")


def main():
    print("=" * 65)
    print("LAB 09 — Information Extraction Application (Option A: News)")
    print("=" * 65)

    extractor = NewsInformationExtractor()

    # 1. Trực quan hóa pipeline trên các ví dụ văn bản thô
    sample_texts = [
        "Barack Obama visited Paris yesterday to attend the climate conference.",
        "Elon Musk announced that SpaceX successfully completed tests in California.",
        "Microsoft and OpenAI formed a major partnership based in San Francisco.",
        "The United Nations Security Council convened an emergency meeting in New York."
    ]

    print("\n--- DEMO: RAW TEXT -> STRUCTURED INFORMATION ---")
    for text in sample_texts:
        structured = extractor.extract_entities(text)
        print(f"\nRaw Text : \"{text}\"")
        print("Structured Output (JSON):")
        print(json.dumps(structured, indent=2, ensure_ascii=False))

    # 2. Đánh giá định lượng trên 30 câu kiểm thử
    evaluate_ie_pipeline(extractor, num_sentences=30)

if __name__ == "__main__":
    main()
