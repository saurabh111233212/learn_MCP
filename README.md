# Learning MCP (Model Context Protocol)

This project demonstrates how to build an application using the Model Context Protocol (MCP), featuring a combined server providing multiple capabilities and a client chatbot that utilizes them.

It uses the official [mcp-python](https://github.com/modelcontextprotocol/python-sdk) SDK and the `FastMCP` interface for creating the server.

## What's Included

*   `combined_mcp_server.py`: An MCP server using `FastMCP` that provides:
    *   **Resources**: System context (`context://system`) and calculator documentation (`info://calculator`).
    *   **Tools**: A calculator tool (`calculator`).
    *   **Prompts**: A text summarization prompt (`summarize_text`).
*   `combined_mcp_client.py`: An interactive chatbot client that:
    *   Connects to the server via `stdio`.
    *   Fetches and uses system context.
    *   Integrates with OpenAI, passing available tools (calculator).
    *   Handles tool execution requests from the LLM.
    *   Allows listing and using server-defined prompts (e.g., `/list_prompts`, `/use_prompt summarize_text text="..."`).
*   `requirements.txt`: Python dependencies.
*   `.env`: For your OpenAI API key.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure OpenAI:**
    *   Create a `.env` file in the project root.
    *   Add your OpenAI API key:
        ```env
        OPENAI_API_KEY="your-api-key-here"
        ```

## Running the Combined Demo

The client script runs the server script via the `stdio` transport.

```bash
# Run the client (it will automatically start the server)
python combined_mcp_client.py 
```

Once running, you can:
*   Chat normally with the LLM.
*   Ask it to perform calculations (e.g., "What is 5 times 12?").
*   Use commands:
    *   `/list_tools`
    *   `/list_prompts`
    *   `/use_prompt summarize_text text="Your long text here"`
*   Type `quit` to exit.

## Understanding MCP Concepts Used

*   **FastMCP Server**: High-level server definition using decorators (`@mcp_server.resource`, `@mcp_server.tool`, `@mcp_server.prompt`).
*   **Resource URIs**: Standardized way to identify context data (e.g., `context://system`).
*   **Tool Integration**: Defining tools on the server and passing them to the LLM via the client for execution.
*   **Prompt Templates**: Defining reusable interaction patterns on the server (`@mcp_server.prompt`) and invoking them from the client (`session.get_prompt`).
*   **ClientSession / stdio Transport**: Managing the client-server connection and communication locally.

## Project Structure

```
learn_MCP/
├── combined_mcp_server.py    # Combined server definition (resources, tools, prompts)
├── combined_mcp_client.py    # Chatbot client (runs server via stdio)
├── requirements.txt          # Python dependencies
├── .env                      # API key configuration
└── README.md                 # This file
```

## Learning More

*   [MCP Documentation](https://modelcontextprotocol.io/)
*   [MCP Python SDK README](https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md)
*   [MCP Specification](https://spec.modelcontextprotocol.io/)
