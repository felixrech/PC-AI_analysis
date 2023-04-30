"""Module that provides the topic details page and its callbacks."""

import dash

from layouts import topic_details as layout_
from callbacks import topic_details as callbacks


# Register page for multi-page setup
dash.register_page(__name__, path_template="/topic_details/")


layout = layout_.layout


callbacks.setup_callbacks()
