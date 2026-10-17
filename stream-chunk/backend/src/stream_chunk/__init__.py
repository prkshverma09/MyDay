"""HTTP API that turns closed video uploads into timed MP4 chunks."""

__all__ = ["create_app"]


def create_app():
    from stream_chunk.app import create_app as build_app

    return build_app()
