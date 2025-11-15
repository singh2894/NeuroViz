import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_quick_model(df: pl.DataFrame, target: str):
    pdf = df.to_pandas()
    y = pdf[target]
    X = pdf.drop(columns=[target])
    num = X.select_dtypes(include=["number"]).columns
    cat = X.select_dtypes(exclude=["number"]).columns
    pre = ColumnTransformer(
        [
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    model = Pipeline(
        [("pre", pre), ("rf", RandomForestClassifier(n_estimators=200, random_state=0))]
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model.fit(Xtr, ytr)
    return model, Xte, yte
