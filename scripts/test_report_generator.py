from src.utils.report_generator import generate_report
from src.data.dataset import DEFAULT_LABELS

probs = [0.42, 0.45, 0.32, 0.64, 0.43]

report = generate_report(DEFAULT_LABELS, probs, threshold=0.5)
print(report)