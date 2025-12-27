import torch


def f1_micro(
    logits,
    labels,
    threshold=0.5,
    label_mask=None,
):
    preds = torch.sigmoid(logits) > threshold
    if label_mask is not None:
        mask = label_mask > 0
        preds = preds[mask]
        labels = labels[mask]
        if preds.numel() == 0:
            return 0.0
    tp = (preds & (labels > 0)).sum().item()
    fp = (preds & (labels == 0)).sum().item()
    fn = (~preds & (labels > 0)).sum().item()
    denom = tp + 0.5 * (fp + fn)
    return tp / denom if denom > 0 else 0.0


def f1_macro(
    logits,
    labels,
    threshold=0.5,
    label_mask=None,
):
    preds = torch.sigmoid(logits) > threshold
    scores = []
    for i in range(labels.size(1)):
        if label_mask is not None:
            m = label_mask[:, i] > 0
            if not m.any():
                continue
            l = labels[m, i] > 0
            p = preds[m, i]
        else:
            l = labels[:, i] > 0
            p = preds[:, i]
        tp = (p & l).sum().item()
        fp = (p & ~l).sum().item()
        fn = (~p & l).sum().item()
        denom = tp + 0.5 * (fp + fn)
        scores.append(tp / denom if denom > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
