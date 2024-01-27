def overlap_coefficient(A: set, B: set) -> float:
    len_A, len_B = len(A), len(B)

    return len(A & B) / min(len(A), len(B)) if len_A and len_B else 0.0
