from functools import cache

import pandas as pd
from datasets import load_dataset


@cache
def load_multilingual_question_dataset(
    languages_subset: set[str] | None = None,
    min_length: int = 30,
    max_length: int = 200,
) -> pd.DataFrame:
    # df_fquad = load_dataset("illuin/fquad", trust_remote_code=True)["train"].to_pandas()
    df_aya = load_dataset("CohereForAI/aya_dataset", "default")["train"].to_pandas()
    df_xquad_de = load_dataset("google/xquad", "xquad.de")["validation"].to_pandas()
    df_xquad_th = load_dataset("google/xquad", "xquad.th")["validation"].to_pandas()
    df_squad_it = load_dataset("crux82/squad_it")["train"].to_pandas()
    df_mlqa_hi = load_dataset(
        "facebook/mlqa", "mlqa-translate-train.hi", trust_remote_code=True
    )["train"].to_pandas()

    df_aya = df_aya[["inputs", "language"]].rename(columns={"inputs": "question"})
    df_xquad_de = df_xquad_de[["question"]].assign(language="German")
    df_xquad_th = df_xquad_th[["question"]].assign(language="Thai")
    df_squad_it = df_squad_it[["question"]].assign(language="Italian")
    df_mlqa_hi = df_mlqa_hi[["question"]].assign(language="Hindi")

    df = pd.concat([df_aya, df_xquad_de, df_xquad_th, df_squad_it, df_mlqa_hi])

    if languages_subset:
        df = df[df["language"].isin(languages_subset)]

    # filter out bad samples from AYA dataset
    df = df.drop_duplicates("question")

    df["length"] = df["question"].map(len)
    df = df[(df["length"] > min_length) & (df["length"] < max_length)]

    # balance dataset by taking the same number of samples for each language
    min_language_count = df["language"].value_counts().min()
    df = df.sort_values("length").groupby("language").head(min_language_count)

    df["id"] = df.groupby("language").cumcount()
    return df.pivot(index="id", values="question", columns="language")
