#!/usr/bin/env python3
import logging
import asyncio
import math
from mcp.server.fastmcp import FastMCP
from mcp.model import TextContent, SamplingMessage, SamplingRole # Needed for prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp_server = FastMCP(
    name="Combined MCP Server",
    description="Provides context resources, a calculator tool, and prompt templates."
)

# --- Resources --- 

@mcp_server.resource("context://system")
def get_system_context() -> str:
    """Provides the main system prompt/context for the LLM."""
    logger.info("Providing system context resource")
    return ("You are a helpful AI assistant integrated via the Model Context Protocol (MCP). "
            "You have access to context resources and can use available tools like a calculator.")

@mcp_server.resource("info://calculator")
def get_calculator_info() -> str:
    """Provides documentation for the calculator tool."""
    logger.info("Providing calculator info resource")
    return """# Calculator Tool

Use the 'calculator' tool to perform mathematical operations:
- `add`, `subtract`, `multiply`, `divide`, `power` (requires `a` and `b`)
- `sqrt`, `sin`, `cos`, `tan` (requires `a` only)

Example LLM call: `calculator(operation="multiply", a=7, b=6)`
"""

# --- Tools --- 

@mcp_server.tool()
def calculator(operation: str, a: float, b: float = None) -> float:
    """Perform mathematical calculations.

    Args:
        operation: The mathematical operation (add, subtract, multiply, divide, power, sqrt, sin, cos, tan).
        a: First operand.
        b: Second operand (required for binary operations).
    """
    logger.info(f"Executing calculator tool: operation={operation}, a={a}, b={b}")
    
    binary_ops = ["add", "subtract", "multiply", "divide", "power"]
    unary_ops = ["sqrt", "sin", "cos", "tan"]

    if operation in binary_ops and b is None:
        raise ValueError(f"Parameter 'b' is required for operation '{operation}'")
    if operation not in binary_ops and operation not in unary_ops:
         raise ValueError(f"Unknown operation: {operation}")

    if operation == "add": return a + b
    if operation == "subtract": return a - b
    if operation == "multiply": return a * b
    if operation == "divide":
        if b == 0: raise ValueError("Division by zero")
        return a / b
    if operation == "power": return a ** b
    if operation == "sqrt":
        if a < 0: raise ValueError("Cannot take square root of negative number")
        return math.sqrt(a)
    if operation == "sin": return math.sin(a)
    if operation == "cos": return math.cos(a)
    if operation == "tan": return math.tan(a)
    
    # Should be unreachable due to check above, but included for safety
    raise ValueError(f"Unhandled operation: {operation}") 

# --- Prompts --- 

@mcp_server.prompt()
def summarize_text(text: str) -> list[SamplingMessage]:
    """Creates a prompt asking the LLM to summarize the provided text."""
    logger.info(f"Creating summarize_text prompt for text: {text[:50]}...")
    return [
        SamplingMessage(role=SamplingRole.USER, content=TextContent(text=f"Please summarize the following text concisely:\n\n{text}"))
    ]

# --- Server Definition Complete --- 

logger.info("Combined MCP Server defined. Ready to be run via stdio_client.")

# Note: No main execution block needed here when using stdio_client.
# The client process will run this script. 