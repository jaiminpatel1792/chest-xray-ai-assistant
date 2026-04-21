from typing import Dict, List
from typing import Optional


LABEL_DESCRIPTIONS = {
    "Atelectasis": "partial collapse or reduced expansion of lung tissue",
    "Cardiomegaly": "enlargement of the heart silhouette",
    "Consolidation": "lung airspaces filled with fluid, pus, or other material",
    "Edema": "fluid accumulation in the lungs",
    "Pleural Effusion": "fluid collection in the pleural space around the lungs",
}


def format_probabilities(labels: List[str], probs: List[float]) -> Dict[str, float]:
    return {label: float(prob) for label, prob in zip(labels, probs)}


def get_positive_findings(prob_map: Dict[str, float], threshold: float = 0.5) -> Dict[str, float]:
    return {label: prob for label, prob in prob_map.items() if prob >= threshold}


def get_ground_truth_positive(labels: List[str], gt: List[float]) -> List[str]:
    return [label for label, val in zip(labels, gt) if int(val) == 1]


def generate_report(
    labels: List[str],
    probs: List[float],
    ground_truth: Optional[List[float]] = None,
    threshold: float = 0.5,
) -> str:
    prob_map = format_probabilities(labels, probs)
    positives = get_positive_findings(prob_map, threshold=threshold)

    top_label = max(prob_map, key=prob_map.get)
    top_conf = prob_map[top_label]

    gt_positive = []
    if ground_truth is not None:
        gt_positive = get_ground_truth_positive(labels, ground_truth)

    findings_lines = []
    if positives:
        sorted_findings = sorted(positives.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_findings:
            desc = LABEL_DESCRIPTIONS.get(label, label)
            findings_lines.append(f"- {label} ({prob:.3f}): suggests {desc}.")
    else:
        findings_lines.append(
            "- No pathology exceeded the selected confidence threshold."
        )

    findings_text = "Findings:\n" + "\n".join(findings_lines)

    impression_text = (
        f"\n\nImpression: The highest-confidence model prediction is {top_label} "
        f"with probability {top_conf:.3f}."
    )

    if gt_positive:
        impression_text += (
            "\nGround truth positive labels: " + ", ".join(gt_positive) + "."
        )
    elif ground_truth is not None:
        impression_text += "\nGround truth positive labels: none."

    if ground_truth is not None:
        pred_binary = [1 if p >= threshold else 0 for p in probs]
        gt_binary = [int(v) for v in ground_truth]
        match = pred_binary == gt_binary
        impression_text += (
            "\nPrediction summary: "
            + ("matches ground truth at threshold 0.5." if match else "does not fully match ground truth at threshold 0.5.")
        )

    disclaimer = (
        "\n\nDisclaimer: This is a model-generated research summary for educational "
        "and experimental use only. It is not a clinical diagnosis."
    )

    return findings_text + impression_text + disclaimer