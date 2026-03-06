import torch.nn as nn

def build_loss(pos_weight):
    criterion_p = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_r = nn.CrossEntropyLoss()
    return criterion_p, criterion_r