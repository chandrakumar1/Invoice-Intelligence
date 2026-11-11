"""
What:     Marks `app` as a package and records the pipeline version stamped into every result.
In:       Nothing.
Out:      __version__, read by the pipeline when building an ExtractionResult.
Fails if: The version isn't bumped when output semantics change, so stored results can't be told apart.
"""

__version__ = "0.1.0"
