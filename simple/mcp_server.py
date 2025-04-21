#!/usr/bin/env python3
import logging
import asyncio
from mcp.server.fastmcp import FastMCP # Use FastMCP
from mcp.server.stdio import stdio_server # Import stdio_server
# We don't need ResourceProvider or manual Resource imports with FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp_server = FastMCP(
    name="Simple MCP Context Provider",
    description="Provides system context and information about MCP to LLMs"
)

# Define the system context as a resource using the decorator
@mcp_server.resource("context://system")
def get_system_context() -> str:
    """Provides the main system prompt/context."""
    logger.info("Providing system context resource")
    return "You are a helpful AI assistant integrated through the Model Context Protocol (MCP). You have access to contextual information provided by the MCP server."

# Define the MCP info as another resource
@mcp_server.resource("info://mcp")
def get_mcp_info() -> str:
    """Provides informational text about MCP."""
    logger.info("Providing MCP info resource")
    return ("# Model Context Protocol\n\n" 
            "MCP is an open protocol that standardizes how applications provide context to LLMs. "
            "It defines methods for sharing contextual information, tools, and capabilities between hosts and servers.")

async def main():
    """Run the server using stdio transport."""
    logger.info("Starting MCP server via stdio...")
    async with stdio_server(mcp_server):
        # Keep the server running
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())

# Optional: Add a way to run it standalone for debugging if needed, 
# but the primary use case here is via stdio.
# if __name__ == "__main__":
#    # Requires additional setup to run FastMCP standalone (e.g., with uvicorn)
#    print("To run this server, use 'mcp dev real_mcp_server.py' or connect via stdio_client.")
#    pass 