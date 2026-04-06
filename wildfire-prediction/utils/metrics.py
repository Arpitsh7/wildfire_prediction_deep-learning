import torch


def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = targets.float()
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = float(intersection / (union + 1e-8))
    
    dice = float(2 * intersection / (preds.sum() + targets.sum() + 1e-8))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.iou = 0
        self.dice = 0
        self.count = 0
    
    def update(self, preds, targets, threshold=0.5):
        metrics = compute_metrics(preds, targets, threshold)
        self.precision += metrics['precision']
        self.recall += metrics['recall']
        self.f1 += metrics['f1']
        self.iou += metrics['iou']
        self.dice += metrics['dice']
        self.count += 1
    
    def get_avg(self):
        if self.count == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'iou': 0,
                'dice': 0
            }
        return {
            'precision': self.precision / self.count,
            'recall': self.recall / self.count,
            'f1': self.f1 / self.count,
            'iou': self.iou / self.count,
            'dice': self.dice / self.count
        }
