"""Module providing the TSNE embedding page."""

import dash

from layouts import embedding as layout_
from callbacks import embedding as callbacks


# Register page for multi-page setup
dash.register_page(__name__, path_template="/embedding/")


layout = layout_.layout


callbacks.setup_callbacks()
