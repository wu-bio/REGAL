"""Experimental RSS calculation package."""

__all__ = ["RSSPipelineConfig", "RSS_OUTPUT_DIR", "run_rss_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import RSSPipelineConfig, RSS_OUTPUT_DIR, run_rss_pipeline

        exports = {
            "RSSPipelineConfig": RSSPipelineConfig,
            "RSS_OUTPUT_DIR": RSS_OUTPUT_DIR,
            "run_rss_pipeline": run_rss_pipeline,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
