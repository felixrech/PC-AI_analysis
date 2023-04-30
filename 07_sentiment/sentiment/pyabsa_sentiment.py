import warnings
import pandas as pd

from pyabsa import AspectPolarityClassification as APC


sentiment_classifier = APC.SentimentClassifier(checkpoint="english")

# Data wrangling: handle input
df = (
    pd.read_feather("/home/felix/uni/thesis/06_explorer/explorer/data/sentiments.arrow")
    .drop(columns=["negative", "neutral", "positive", "sentiment"])
    .dropna(subset="aspect")
)
sentiment_string = (
    df["sentence_before"] + "[B-ASP]" + df["aspect"] + "[E-ASP]" + df["sentence_after"]
)

# Compute ABSA for each mention
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = list(
        map(
            lambda x: sentiment_classifier.predict(text=x, print_result=False),
            sentiment_string.to_list(),
        )
    )

# Batch processing would be a lot faster but looses a few mentions somewhere in between
# (without throwing an exception/error)
# results = sentiment_classifier.predict(
#     text=sentiment_string.to_list(),
#     print_result=False,
#     # ignore_error=True,
#     eval_batch_size=32,
# )


# Data wrangle output into usable format
results_raw = list(map(lambda x: x["probs"][0], results))
results_sent = list(map(lambda x: x["sentiment"][0].lower(), results))
results = pd.concat(
    (
        df.reset_index(drop=True),
        pd.DataFrame(results_raw, columns=["negative", "neutral", "positive"]),
        pd.DataFrame(results_sent, columns=["sentiment"]),
    ),
    axis=1,
)

# Save output (only two columns to save space)
results[["sentence_index", "sentiment"]].to_feather(
    "../../06_explorer/explorer/data/pyabsa.arrow"
)
