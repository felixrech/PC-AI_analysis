"""Module that set up the /search/ page."""

import dash

from layouts import search as layout_
from utils import sentiment as sentiment_utils
from callbacks import search as callbacks


# Register page for multi-page setup
dash.register_page(__name__, path_template="/search/")


# Load the dataset
sentiments = sentiment_utils.load_sentiment()


layout = layout_.get_layout(sentiments)
callbacks.setup_callbacks()
