from mcp.server.mcpserver import MCPServer

from .config import SERVER_INSTRUCTIONS


mcp: MCPServer = MCPServer(
    name="mcp_server_photomol",
    instructions=SERVER_INSTRUCTIONS,
)

# Import modules for their registration side effects.
# These must come after `mcp` is created.
from .resources import *  # noqa: E402,F403
from .tools import *  # noqa: E402,F403