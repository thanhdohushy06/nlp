"""
baseline_ner.py — Baseline Named Entity Recognition on CoNLL-2003
Using Handcrafted Token Features + Linear Classifier (Logistic Regression / SGDClassifier)
"""

import sys
import numpy as np
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

LABEL_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def word2features(sent, i):
    """
    Trích xuất đặc trưng bề mặt từ vựng cho token thứ i trong câu.
    Bao gồm hình thái từ, tiền tố, hậu tố, cờ viết hoa, số và ngữ cảnh lân cận (cửa sổ [-1, +1]).
    """
    word = str(sent[i])
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
    }
    
    # Đặc trưng từ đứng trước (Previous Word)
    if i > 0:
        prev_word = str(sent[i-1])
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
        
    # Đặc trưng từ đứng sau (Next Word)
    if i < len(sent) - 1:
        next_word = str(sent[i+1])
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def main():
    print("=" * 60)
    print("LAB 09 — Baseline NER (Handcrafted Features + Linear Classifier)")
    print("=" * 60)
    
    # 1. Tải tập dữ liệu CoNLL-2003
    print("\n[1/4] Đang tải CoNLL-2003 dataset...")
    ds = load_dataset('lhoestq/conll2003')
    
    train_sents = ds['train']
    test_sents = ds['test']
    print(f"Số lượng câu Train: {len(train_sents)}, Test: {len(test_sents)}")
    
    # 2. Trích xuất đặc trưng
    print("\n[2/4] Trích xuất đặc trưng token cho tập Train và Test...")
    X_train_raw = [sent2features(s['tokens']) for s in train_sents]
    y_train_raw = [[LABEL_NAMES[t] for t in s['ner_tags']] for s in train_sents]
    
    X_test_raw = [sent2features(s['tokens']) for s in test_sents]
    y_test_raw = [[LABEL_NAMES[t] for t in s['ner_tags']] for s in test_sents]
    
    # Trải phẳng danh sách token để huấn luyện scikit-learn classifier
    X_train_flat = [feat for s in X_train_raw for feat in s]
    y_train_flat = [lbl for s in y_train_raw for lbl in s]
    
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_flat)
    # Ép kiểu chỉ số ma trận thưa sang int32 cho tương thích scikit-learn
    X_train.indices = X_train.indices.astype(np.int32)
    X_train.indptr = X_train.indptr.astype(np.int32)
    
    print(f"Số chiều không gian đặc trưng (Feature dimension): {X_train.shape[1]}")
    
    # 3. Huấn luyện mô hình Linear Classifier (Logistic Regression qua SGDClassifier)
    print("\n[3/4] Huấn luyện Linear Classifier (SGD with log_loss)...")
    clf = SGDClassifier(loss='log_loss', max_iter=30, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_flat)
    print("Huấn luyện hoàn tất!")
    
    # 4. Dự đoán và Đánh giá trên tập Test
    print("\n[4/4] Dự đoán và Đánh giá Entity-Level trên tập Test...")
    X_test_flat = [feat for s in X_test_raw for feat in s]
    X_test = vec.transform(X_test_flat)
    X_test.indices = X_test.indices.astype(np.int32)
    X_test.indptr = X_test.indptr.astype(np.int32)
    
    preds_flat = clf.predict(X_test)
    
    # Tái cấu trúc dự đoán về dạng danh sách câu
    preds = []
    idx = 0
    for s in y_test_raw:
        preds.append(list(preds_flat[idx:idx + len(s)]))
        idx += len(s)
        
    p = precision_score(y_test_raw, preds)
    r = recall_score(y_test_raw, preds)
    f1 = f1_score(y_test_raw, preds)
    
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ BASELINE (Entity-Level):")
    print(f"  - Precision : {p:.4f} ({p*100:.2f}%)")
    print(f"  - Recall    : {r:.4f} ({r*100:.2f}%)")
    print(f"  - F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 60)
    print("\nBáo cáo chi tiết theo từng loại thực thể (seqeval classification report):")
    print(classification_report(y_test_raw, preds))

if __name__ == "__main__":
    main()
