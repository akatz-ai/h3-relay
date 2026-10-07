"""ComfyUI entrypoint for H3 Relay."""

from .h3_relay import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .h3_relay.staged import register_routes
from .h3_relay.removal_previews import register_routes as register_removal_routes

WEB_DIRECTORY = "./web"
register_routes()
register_removal_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
