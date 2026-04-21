from src.utils.report_generator import generate_report


def build_report(labels, probs, ground_truth=None, threshold=0.5):
    return generate_report(
        labels=labels,
        probs=probs,
        ground_truth=ground_truth,
        threshold=threshold,
    )