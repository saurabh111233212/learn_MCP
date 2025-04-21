#!/usr/bin/env python3
import os
import logging
import asyncio
import sys
from mcp import (
    ClientSession, 
    StdioServerParameters, 
    # Import specific types if needed, e.g., for listing
    Tool # For inspecting tool specs if necessary
)
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define server parameters for stdio connection
server_params = StdioServerParameters(
    command=sys.executable,  # Use the current Python interpreter
    args=["mcp_tool_server.py"], # Argument is the server script
)

class CalculatorClient:
    """
    An MCP client connecting via stdio to use the calculator tool.
    """
    def __init__(self):
        pass # No client object needed until run_session

    async def run_session(self):
        """Connect to server, run interactive calculator session."""
        logger.info("Starting stdio client for calculator server...")
        try:
            async with stdio_client(server_params) as (read, write):
                logger.info("stdio transport connected, creating client session.")
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session...")
                    server_info = await session.initialize()
                    logger.info(f"Connected to server: {server_info.server_name}")

                    # Optionally list tools to verify
                    try:
                        tools = await session.list_tools()
                        logger.info(f"Available tools: {[t.name for t in tools]}")
                        if not any(t.name == 'calculator' for t in tools):
                            print("Error: Calculator tool not found on server.")
                            return
                    except Exception as e:
                        logger.error(f"Failed to list tools: {e}")
                        print("Error listing tools from server.")
                        return
                    
                    # Get and display calculator info resource
                    print("\nCalculator Tool Documentation:")
                    print("-----------------------------")
                    try:
                        content, mime_type = await session.read_resource("info://calculator")
                        info_text = content.decode('utf-8') if isinstance(content, bytes) else str(content)
                        print(info_text)
                    except Exception as e:
                         logger.error(f"Failed to get calculator info: {e}")
                         print("[Could not load documentation]")
                         
                    print("\nInteractive Calculator (type 'quit' to exit)")
                    print("-------------------------------------------")
                    
                    while True:
                        operation = await asyncio.to_thread(input, "\nOperation (or 'quit'): ")
                        operation = operation.lower().strip()
                        if operation == 'quit':
                            print("Exiting calculator.")
                            break
                        if not operation:
                            continue

                        try:
                            a_str = await asyncio.to_thread(input, "Enter first number (a): ")
                            a = float(a_str)
                            
                            b = None
                            if operation not in ['sqrt', 'sin', 'cos', 'tan']:
                                b_str = await asyncio.to_thread(input, "Enter second number (b): ")
                                b = float(b_str)
                                
                            # Prepare parameters for the tool call
                            params = {"operation": operation, "a": a}
                            if b is not None:
                                params["b"] = b
                            
                            # Call the tool
                            logger.info(f"Calling calculator tool with params: {params}")
                            result = await session.call_tool("calculator", arguments=params)
                            print(f"\nResult: {result}") # Assuming result is directly usable
                            
                        except ValueError:
                            print("Invalid number entered. Please try again.")
                        except Exception as e:
                            # Catch errors from tool execution (e.g., division by zero)
                            logger.error(f"Tool call failed: {e}", exc_info=True)
                            print(f"Error: {e}")
                            
        except ConnectionRefusedError:
            logger.error("Connection refused. Is the server script available and correct?")
        except Exception as e:
            logger.error(f"Client session error: {e}", exc_info=True)

async def main():
    client = CalculatorClient()
    await client.run_session()

if __name__ == "__main__":
    asyncio.run(main()) 