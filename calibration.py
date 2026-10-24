"""Coordinate-offset calibration; no model weights are modified."""


def average_offsets(data: dict) -> dict:
    """Calculate average offset for each landmark."""
    offsets = {}
    for name, info in data.items():
        count = info.get("count", 0)
        if count:
            sx, sy = info.get("sum", [0, 0])
            offsets[name] = (sx / count, sy / count)
    return offsets

def update_training(data: dict, predicted: dict, corrected: dict) -> None:
    """Update running sums based on user corrections."""
    for name, pred_pt in predicted.items():
        corr_pt = corrected.get(name, pred_pt)
        diff_x = corr_pt[0] - pred_pt[0]
        diff_y = corr_pt[1] - pred_pt[1]
        if name in data:
            data[name]["sum"][0] += diff_x
            data[name]["sum"][1] += diff_y
            data[name]["count"] += 1
        else:
            data[name] = {"sum": [diff_x, diff_y], "count": 1}
