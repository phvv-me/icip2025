import pandas as pd
import yaml


def load_models(filepath="models.yaml"):
    return pd.DataFrame(yaml.safe_load(open(filepath))).fillna(value="auto")
