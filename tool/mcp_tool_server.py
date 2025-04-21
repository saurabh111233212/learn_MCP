#!/usr/bin/env python3
import logging
import asyncio
import math
from mcp.server.fastmcp import FastMCP # Use FastMCP
# Types might be needed for tool signatures if FastMCP infers them
# from mcp import ToolSpec # Likely not needed with decorators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp_server = FastMCP(
    name="MCP Calculator Server",
    description="Provides a calculator tool and documentation via MCP"
)

# Define the calculator tool using the decorator
# Type hints in the function signature define the tool parameters
@mcp_server.tool()
def calculator(operation: str, a: float, b: float = None) -> float:
    """Perform mathematical calculations.

    Args:
        operation: The mathematical operation to perform (add, subtract, multiply, divide, power, sqrt, sin, cos, tan).
        a: First operand.
        b: Second operand (not required for sqrt, sin, cos, tan).
    """
    logger.info(f"Executing calculator tool: operation={operation}, a={a}, b={b}")
    
    if operation == "add":
        if b is None: raise ValueError("Parameter 'b' required for add")
        return a + b
    elif operation == "subtract":
        if b is None: raise ValueError("Parameter 'b' required for subtract")
        return a - b
    elif operation == "multiply":
        if b is None: raise ValueError("Parameter 'b' required for multiply")
        return a * b
    elif operation == "divide":
        if b is None: raise ValueError("Parameter 'b' required for divide")
        if b == 0: raise ValueError("Division by zero")
        return a / b
    elif operation == "power":
        if b is None: raise ValueError("Parameter 'b' required for power")
        return a ** b
    elif operation == "sqrt":
        if a < 0: raise ValueError("Cannot take square root of negative number")
        return math.sqrt(a)
    elif operation == "sin":
        return math.sin(a)
    elif operation == "cos":
        return math.cos(a)
    elif operation == "tan":
        return math.tan(a)
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Define the calculator documentation as a resource
@mcp_server.resource("info://calculator")
def get_calculator_info() -> str:
    """Provides documentation for the calculator tool."""
    logger.info("Providing calculator info resource")
    # Use a standard triple-quoted string for multiline documentation
    return """# Calculator Tool

This MCP server provides a calculator tool that can perform various mathematical operations:

- `add`: Add two numbers (a + b)
- `subtract`: Subtract second number from first (a - b)
- `multiply`: Multiply two numbers (a * b)
- `divide`: Divide first number by second (a / b)
- `power`: Raise first number to the power of second (a ^ b)
- `sqrt`: Square root of a number (√a)
- `sin`: Sine of an angle in radians
- `cos`: Cosine of an angle in radians
- `tan`: Tangent of an angle in radians

Example usage with an LLM might involve asking it to call:
`calculator(operation="add", a=5, b=3)`
"""

# This script just defines the server; it's run via stdio_client.
logger.info("MCP Calculator Server defined. Ready to be run via stdio_client.") 