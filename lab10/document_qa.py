"""
document_qa.py — Mini Document QA Application

Ứng dụng extractive QA trên documents.
Cho phép user nhập question và context, trả về answer span.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def load_qa_pipeline(model_name="deepset/bert-base-cased-squad2"):
    """Load pretrained QA model và tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def answer_question(question, context, model, tokenizer, max_length=384):
    """
    Trả lời câu hỏi dựa trên context cho trước.

    Args:
        question: câu hỏi
        context: đoạn văn chứa câu trả lời
        model: QA model
        tokenizer: tokenizer
        max_length: max sequence length

    Returns:
        dict với 'answer', 'score', 'start', 'end'
    """
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    best_answer = {"answer": "", "score": float("-inf"), "start": 0, "end": 0}

    with torch.no_grad():
        for i in range(len(inputs["input_ids"])):
            model_inputs = {
                "input_ids": inputs["input_ids"][i].unsqueeze(0),
                "attention_mask": inputs["attention_mask"][i].unsqueeze(0),
            }
            if "token_type_ids" in inputs:
                model_inputs["token_type_ids"] = inputs["token_type_ids"][i].unsqueeze(0)

            outputs = model(**model_inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            offsets = inputs["offset_mapping"][i]
            seq_ids = inputs.sequence_ids(i)

            # Chỉ xét context tokens
            start_logits_masked = start_logits.clone()
            end_logits_masked = end_logits.clone()
            for idx, sid in enumerate(seq_ids):
                if sid != 1:
                    start_logits_masked[idx] = float("-inf")
                    end_logits_masked[idx] = float("-inf")

            # Tìm top start/end
            start_idx = torch.argmax(start_logits_masked).item()
            end_idx = torch.argmax(end_logits_masked).item()

            if end_idx < start_idx:
                end_idx = start_idx

            score = start_logits[start_idx].item() + end_logits[end_idx].item()

            if score > best_answer["score"]:
                # Lấy answer text từ context bằng offset mapping
                start_char = offsets[start_idx][0].item()
                end_char = offsets[end_idx][1].item()
                answer_text = context[start_char:end_char]

                best_answer = {
                    "answer": answer_text,
                    "score": score,
                    "start": start_char,
                    "end": end_char
                }

    return best_answer


def document_qa_demo():
    """Demo interactive Document QA."""
    print("=" * 60)
    print("  Document QA — Extractive Question Answering")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_qa_pipeline()
    print("Model loaded!\n")

    # Demo documents
    documents = {
        "Wikipedia - Python": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use of significant indentation. "
            "Python is dynamically typed and garbage-collected. "
            "It supports multiple programming paradigms, including structured, object-oriented and functional programming. "
            "It was created by Guido van Rossum and first released in 1991."
        ),
        "Wikipedia - Machine Learning": (
            "Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being explicitly programmed. "
            "Machine learning focuses on the development of computer programs that can access data "
            "and use it to learn for themselves. The process of learning begins with observations or data, "
            "such as examples, direct experience, or instruction, in order to look for patterns in data "
            "and make better decisions in the future based on the examples that we provide."
        ),
        "FAQ - University": (
            "The tuition fee for undergraduate students is $5,000 per semester. "
            "Graduate students pay $7,500 per semester. "
            "The refund policy allows full refund within the first two weeks of the semester. "
            "After two weeks, only 50% refund is available. "
            "No refund is given after the fourth week. "
            "Students can apply for financial aid through the Student Affairs Office."
        )
    }

    print("Available documents:")
    for i, name in enumerate(documents.keys(), 1):
        print(f"  {i}. {name}")

    print("\n--- Demo Questions ---\n")

    demo_questions = [
        ("Wikipedia - Python", "Who created Python?"),
        ("Wikipedia - Python", "When was Python first released?"),
        ("Wikipedia - Machine Learning", "What is machine learning?"),
        ("FAQ - University", "What is the tuition fee for graduate students?"),
        ("FAQ - University", "What is the refund policy?"),
    ]

    for doc_name, question in demo_questions:
        context = documents[doc_name]
        result = answer_question(question, context, model, tokenizer)
        print(f"[{doc_name}]")
        print(f"  Q: {question}")
        print(f"  A: {result['answer']} (score: {result['score']:.2f})")
        print()

    # Interactive mode
    print("\n--- Interactive Mode ---")
    print("Enter 'quit' to exit.\n")

    while True:
        print("Choose document (1/2/3) or enter custom context:")
        user_input = input("> ").strip()

        if user_input.lower() == "quit":
            break

        if user_input in ["1", "2", "3"]:
            doc_name = list(documents.keys())[int(user_input) - 1]
            context = documents[doc_name]
            print(f"Using: {doc_name}")
        else:
            context = user_input
            print("Using custom context.")

        question = input("Question: ").strip()
        if question.lower() == "quit":
            break

        result = answer_question(question, context, model, tokenizer)
        print(f"\nAnswer: {result['answer']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Position: [{result['start']}:{result['end']}]\n")


if __name__ == "__main__":
    document_qa_demo()
