def normalize_distance(d, max_d=1.5):
    """
    Converts FAISS distance to similarity between 0 and 1
    Lower distance = stronger evidence
    """
    return max(0, 1 - d / max_d)


def compute_confidence(distances):
    if not distances:
        return 0.0

    similarities = [normalize_distance(d) for d in distances]

    avg_strength = sum(similarities) / len(similarities)

    # coverage reward (more confirming evidence increases trust)
    coverage_factor = min(len(similarities) / 5, 1)

    raw_score = (0.75 * avg_strength + 0.25 * coverage_factor)

    # scale to 0–10 like real confidence metrics
    return round(raw_score * 10, 2)


def confidence_label(score):
    if score >= 9:
        return "Very Strong"
    if score >= 7.5:
        return "Strong"
    if score >= 5.5:
        return "Moderate"
    if score >= 3.5:
        return "Weak"
    return "Very Weak"
