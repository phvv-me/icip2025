import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def save_plot_and_show(name: str, df: pd.DataFrame):
    df.to_pickle(f"resources/{name}.pkl")
    plt.savefig(f"resources/{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def histplot_and_save(name: str, df: pd.DataFrame, x: str, hue: str, *args, **kwargs):
    sns.histplot(
        df, x=x, hue=hue, palette="colorblind", multiple="stack", *args, **kwargs
    )
    save_plot_and_show(name, df)


def lineplot_and_save(
    name: str, df: pd.DataFrame, x: str, y: str, hue: str, *args, **kwargs
):
    sns.lineplot(
        df,
        x=x,
        y=y,
        hue=hue,
        style=hue,
        markers=True,
        dashes=False,
        errorbar=("ci", 99),
        palette="colorblind",
        linewidth=2,
        *args,
        **kwargs,
    )
    plt.xlim(0, df.groupby(hue)[x].max().min())
    save_plot_and_show(name, df)
