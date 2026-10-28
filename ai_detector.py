# =========================================================
# AI TEXT DETECTOR PDF
# AI INDEX VERSION (NON-PERCENTAGE)
# =========================================================

import re
import fitz
import torch
import numpy as np

from collections import Counter
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "gpt2"

MAX_SAMPLE_CHARS = 8000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("LOADING MODEL...")
print("=" * 60)

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)

model.eval()

print(f"Model  : {MODEL_NAME}")
print(f"Device : {DEVICE}")

print("=" * 60)

# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        doc = fitz.open(pdf_path)

        for page in doc:

            page_text = page.get_text()

            text += page_text + "\n"

        doc.close()

    except Exception as e:

        print(f"Error reading PDF: {e}")

    return text

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text

# =========================================================
# SPLIT SENTENCES
# =========================================================

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 10
    ]

# =========================================================
# CALCULATE PERPLEXITY
# =========================================================

def calculate_perplexity(text):

    encodings = tokenizer(
        text,
        return_tensors='pt'
    )

    input_ids = encodings.input_ids.to(DEVICE)

    max_length = model.config.n_positions

    stride = 512

    seq_len = input_ids.size(1)

    nlls = []

    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):

        end_loc = min(
            begin_loc + max_length,
            seq_len
        )

        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[:, begin_loc:end_loc]

        target_ids = input_chunk.clone()

        target_ids[:, :-trg_len] = -100

        with torch.no_grad():

            outputs = model(
                input_chunk,
                labels=target_ids
            )

            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    perplexity = torch.exp(
        torch.stack(nlls).sum() / seq_len
    )

    return perplexity.item()

# =========================================================
# VOCABULARY DIVERSITY
# =========================================================

def vocabulary_diversity(text):

    words = re.findall(
        r'\b\w+\b',
        text.lower()
    )

    total_words = len(words)

    if total_words == 0:
        return 0

    unique_words = len(set(words))

    return unique_words / total_words

# =========================================================
# REPETITION SCORE
# =========================================================

def repetition_score(text):

    words = re.findall(
        r'\b\w+\b',
        text.lower()
    )

    if len(words) == 0:
        return 0

    counts = Counter(words)

    repeated = sum(
        count
        for count in counts.values()
        if count > 3
    )

    return repeated / len(words)

# =========================================================
# BURSTINESS SCORE
# =========================================================

def burstiness_score(text):

    sentences = split_sentences(text)

    if len(sentences) < 2:
        return 0

    lengths = [
        len(sentence.split())
        for sentence in sentences
    ]

    return np.std(lengths)

# =========================================================
# AI INDEX SCORE
# =========================================================

def ai_index_score(
    perplexity,
    vocab_div,
    repetition,
    burstiness
):

    # =====================================================
    # LOWER PERPLEXITY = MORE AI
    # =====================================================

    perplexity_component = max(
        1,
        120 / (perplexity + 1)
    )

    # =====================================================
    # LOWER VOCAB DIVERSITY = MORE AI
    # =====================================================

    vocab_component = (
        (1 - vocab_div) * 40
    )

    # =====================================================
    # HIGHER REPETITION = MORE AI
    # =====================================================

    repetition_component = repetition * 100

    # =====================================================
    # LOWER BURSTINESS = MORE AI
    # =====================================================

    burstiness_component = max(
        0,
        15 - burstiness
    )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    score = (
        perplexity_component
        + vocab_component
        + repetition_component
        + burstiness_component
    )

    return round(score, 0)

# =========================================================
# INTERPRETATION
# =========================================================

def interpret_score(score):

    if score < 20:

        return "Sangat mirip tulisan manusia"

    elif score < 40:

        return "Cenderung tulisan manusia"

    elif score < 70:

        return "Kemungkinan campuran AI dan manusia"

    elif score < 120:

        return "Cenderung AI-generated"

    else:

        return "Sangat kuat AI-generated"

# =========================================================
# ANALYZE PDF
# =========================================================

def analyze_pdf(pdf_path):

    print("\n")
    print("=" * 60)
    print(f"ANALYZING : {pdf_path}")
    print("=" * 60)

    # =====================================================
    # EXTRACT TEXT
    # =====================================================

    text = extract_text_from_pdf(pdf_path)

    if len(text.strip()) == 0:

        print("PDF kosong atau gagal dibaca.")
        return

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    text = clean_text(text)

    # =====================================================
    # LIMIT SAMPLE
    # =====================================================

    text_sample = text[:MAX_SAMPLE_CHARS]

    print(f"Characters analyzed : {len(text_sample)}")

    # =====================================================
    # FEATURES
    # =====================================================

    print("\nCalculating perplexity...")

    perplexity = calculate_perplexity(text_sample)

    vocab_div = vocabulary_diversity(text_sample)

    repetition = repetition_score(text_sample)

    burstiness = burstiness_score(text_sample)

    # =====================================================
    # AI INDEX
    # =====================================================

    ai_score = ai_index_score(
        perplexity,
        vocab_div,
        repetition,
        burstiness
    )

    interpretation = interpret_score(ai_score)

    # =====================================================
    # RESULT
    # =====================================================

    print("\n")
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Perplexity           : {perplexity:.2f}")

    print(f"Vocabulary Diversity : {vocab_div:.4f}")

    print(f"Repetition Score     : {repetition:.4f}")

    print(f"Burstiness Score     : {burstiness:.2f}")

    print("\n" + "-" * 60)

    print(f"AI INDEX SCORE       : {ai_score}")

    print(f"Interpretation       : {interpretation}")

    print("=" * 60)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =============================================
    # CHANGE PDF FILE HERE
    # =============================================

    pdf_file = "ica.pdf"

    analyze_pdf(pdf_file)