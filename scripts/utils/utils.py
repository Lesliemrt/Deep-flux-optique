def normalize(x, minimum, maximum):  # noqa: INP001
    """Normalize x."""
    return (x - minimum) / (maximum - minimum)
