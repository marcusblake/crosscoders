import unicodedata
from typing import Optional
 
import torch
from transformers import AutoModel, AutoTokenizer

def normalize(text: str) -> str:
    """Normalize decoded text for comparison across tokenizers."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2581", " ")   # SentencePiece underscore
    text = text.replace("\u0120", " ")   # GPT-2 Ġ
    return text.lower().strip()
 
 
def is_non_content(token_id: int, tokenizer) -> bool:
    """Return True for whitespace-only or special tokens."""
    if token_id in tokenizer.all_special_ids:
        return True
    decoded = tokenizer.decode([token_id])
    return decoded.strip() == ""

def cross_model_activation_alignment(
    T_A: list[int],
    H_A: torch.Tensor,
    tokenizer_A,
    T_B: list[int],
    H_B: torch.Tensor,
    tokenizer_B,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align activations from two models with potentially different tokenizers.
 
    Args:
        T_A:          Token IDs for Model A  (length N_A)
        H_A:          Activations for Model A, shape (N_A, hidden_dim)
        tokenizer_A:  HuggingFace tokenizer for Model A
        T_B:          Token IDs for Model B  (length N_B)
        H_B:          Activations for Model B, shape (N_B, hidden_dim)
        tokenizer_B:  HuggingFace tokenizer for Model B
 
    Returns:
        H_A_prime:  Aligned activations from Model A, shape (K, hidden_dim_A)
        H_B_prime:  Aligned activations from Model B, shape (K, hidden_dim_B)
        where K is the number of successfully aligned token spans.
    """
    H_A_prime: list[torch.Tensor] = []
    H_B_prime: list[torch.Tensor] = []
 
    p_A, p_B = 0, 0
 
    while p_A < len(T_A) and p_B < len(T_B):
 
        # ── skip non-content tokens ───────────────────────────────────────
        while p_A < len(T_A) and is_non_content(T_A[p_A], tokenizer_A):
            p_A += 1
        while p_B < len(T_B) and is_non_content(T_B[p_B], tokenizer_B):
            p_B += 1
 
        # boundary check after skipping
        if p_A >= len(T_A) or p_B >= len(T_B):
            break
 
        s_A = tokenizer_A.decode([T_A[p_A]])
        s_B = tokenizer_B.decode([T_B[p_B]])
 
        # ── Case 1: simple 1-to-1 match ───────────────────────────────────
        if normalize(s_A) == normalize(s_B):
            H_A_prime.append(H_A[p_A])
            H_B_prime.append(H_B[p_B])
            p_A += 1
            p_B += 1
 
        # ── Case 2: mismatch → window expansion ───────────────────────────
        else:
            e_A, e_B = p_A + 1, p_B + 1
            found_match = False
 
            while e_A <= len(T_A) or e_B <= len(T_B):
                w_A = tokenizer_A.decode(T_A[p_A:e_A])
                w_B = tokenizer_B.decode(T_B[p_B:e_B])
 
                if normalize(w_A) == normalize(w_B):
                    # Take the final token's activation from each window
                    H_A_prime.append(H_A[e_A - 1])
                    H_B_prime.append(H_B[e_B - 1])
                    p_A, p_B = e_A, e_B
                    found_match = True
                    break
 
                # Expand the side with the shorter decoded text
                if len(normalize(w_A)) < len(normalize(w_B)) and e_A <= len(T_A):
                    e_A += 1
                elif e_B <= len(T_B):
                    e_B += 1
                else:
                    break   # cannot expand further
 
            if not found_match:
                # Irreconcilable divergence – return what has been aligned so far
                break
 
    if not H_A_prime:
        # Return empty tensors with correct hidden dims if nothing aligned
        return torch.empty(0, H_A.shape[-1]), torch.empty(0, H_B.shape[-1])
 
    return torch.stack(H_A_prime), torch.stack(H_B_prime)