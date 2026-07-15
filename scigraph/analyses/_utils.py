def generate_p_summary(p: float) -> str:
    """Return Prism-style asterisk notation for a p-value."""
    if p <= 0.0001:
        out = "****"
    elif p <= 0.001:
        out = "***"
    elif p <= 0.01:
        out = "**"
    elif p <= 0.05:
        out = "*"
    else:
        out = "ns"
    return out
