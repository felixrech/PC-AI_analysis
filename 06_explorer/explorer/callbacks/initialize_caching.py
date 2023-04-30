"""Module with utility to initialize the caching of figures, layouts, etc."""

import time


def do() -> None:
    """Initialize the caching of figures, layouts, etc.

    Currently initialized caches:
    - /aspects/ page callbacks (see callbacks/aspects.py for details)
    - /sentiment/ page callbacks (see callbacks/sentiment.py for details)
    """
    # Delayed import needed to ensure dash app is initialized when caching is
    # imported/used
    from callbacks.aspects import initialize_caching as aspects
    from callbacks.overview import initialize_caching as overview
    from callbacks.sentiment import initialize_caching as sentiment
    from callbacks.embedding import initialize_callbacks as embedding
    from callbacks.topic_details import initialize_caching as topic_details
    from callbacks.search import initialize_caching as search

    print("Initializing the caching of callbacks")
    start = time.time()
    overview()
    topic_details()
    embedding()
    aspects()
    sentiment()
    search()
    print(f"Total time: {time.time() - start:.1f}s")
