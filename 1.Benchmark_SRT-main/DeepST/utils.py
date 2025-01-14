from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import pandas as pd

def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()

        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return  ari,nmi,ami