"""Module for the overview page providing some information about a topic model and the
ability to determine similarity with other topic models."""

import dash

from layouts import overview as layout_
from callbacks import overview as callbacks


# Register page for multi-page setup
dash.register_page(__name__, path_template="/overview/")


layout = layout_.layout
callbacks.setup_callbacks()
