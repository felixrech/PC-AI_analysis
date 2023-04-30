"""Module setting up the /aspects/ page."""

import dash

from layouts import aspects as layout_
from callbacks import aspects as callbacks


# Register page for multi-page setup
dash.register_page(__name__, path_template="/aspects/")


layout = layout_.layout
callbacks.setup_callbacks()
