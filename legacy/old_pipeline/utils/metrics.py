from sklearn.metrics import f1_score

def compute_metrics(presence_labels, presence_preds,
                    relation_labels, relation_preds):

    presence_macro_f1 = f1_score(
        presence_labels, presence_preds,
        average="macro", zero_division=0
    )

    relation_macro_f1 = f1_score(
        relation_labels, relation_preds,
        average="macro"
    )

    return presence_macro_f1, relation_macro_f1