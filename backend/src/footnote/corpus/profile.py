"""Shared, deliberately conservative description of the configured corpus.

The datastore is the authority on exact coverage. Agent prompts use this text so adding or
removing documents cannot leave several hard-coded, contradictory corpus descriptions behind.
"""

CORPUS_DESCRIPTION = (
    "a licensed research collection of historical and biographical texts plus open-access "
    "scholarly documents across multiple subjects. Its exact coverage is defined by the "
    "scholars, documents, and passages returned by the corpus tools"
)
