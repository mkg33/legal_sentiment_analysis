import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def attach_predictions(
    df: pd.DataFrame,
    probs,
    emotions,
):
    for i, emo in enumerate(emotions):
        df[f'emo_{emo}'] = probs[:, i]
    return df


def correlate_with_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    emotions,
):
    out = {}
    for emo in emotions:
        col = f'emo_{emo}'
        if col in df and df[outcome_col].nunique() > 1:
            try:
                out[emo] = roc_auc_score(
                    df[outcome_col],
                    df[col],
                )
            except ValueError:
                out[emo] = None
    return out


def regress_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    emotions,
):
    features = df[
        [f'emo_{e}' for e in emotions if f'emo_{e}' in df]
    ].values
    y = df[outcome_col].values
    clf = LogisticRegression(max_iter=200)
    clf.fit(
        features,
        y,
    )
    return clf
