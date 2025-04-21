#!/usr/bin/env python3
import os
import logging
import asyncio
import sys
import json
from mcp import (
    ClientSession, 
    StdioServerParameters,
    # Import types needed
    Resource, 
    Tool,
    Prompt,
    # Need types for OpenAI tool interaction
    ChatCompletionMessage,
    ChatCompletionMessageToolCall
)
from mcp.client.stdio import stdio_client
from openai import OpenAI, APIError # Import APIError for specific handling
from openai.types.chat import ChatCompletion # For type hinting
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence noisy logs from libraries if needed
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Define server parameters for stdio connection
server_params = StdioServerParameters(
    command=sys.executable,  # Use the current Python interpreter
    args=["combined_mcp_server.py"], # Argument is the server script
)

class CombinedMCPChatbot:
    """
    Chatbot using a combined MCP server for context, tools, and prompts.
    """
    def __init__(self):
        try:
            self.openai_client = OpenAI()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}. Check OPENAI_API_KEY.", exc_info=True)
            self.openai_client = None
        self.mcp_session: ClientSession = None
        self.mcp_tools: list[Tool] = []
        self.mcp_prompts: list[Prompt] = []
        self.system_context: str = "You are a helpful assistant." # Default fallback
        self.conversation_history: list[dict] = []

    async def initialize_mcp(self, read, write):
        """Initialize MCP session and fetch initial data."""
        self.mcp_session = ClientSession(read, write)
        logger.info("Initializing MCP session...")
        server_info = await self.mcp_session.initialize()
        logger.info(f"Connected to server: {server_info.server_name}")
        
        # Fetch initial resources, tools, prompts
        await self.fetch_resources()
        await self.fetch_tools()
        await self.fetch_prompts()

    async def fetch_resources(self):
        """Fetch context resources from the server."""
        try:
            logger.info("Fetching system context resource...")
            content, _ = await self.mcp_session.read_resource("context://system")
            self.system_context = content.decode('utf-8') if isinstance(content, bytes) else str(content)
            logger.info("System context updated.")
        except Exception as e:
            logger.warning(f"Could not fetch system context: {e}")
        # Could fetch other resources like calculator info here if needed

    async def fetch_tools(self):
        """Fetch available tools from the server."""
        try:
            logger.info("Fetching tool list...")
            self.mcp_tools = await self.mcp_session.list_tools()
            logger.info(f"Available tools: {[t.name for t in self.mcp_tools]}")
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            self.mcp_tools = []

    async def fetch_prompts(self):
        """Fetch available prompts from the server."""
        try:
            logger.info("Fetching prompt list...")
            self.mcp_prompts = await self.mcp_session.list_prompts()
            logger.info(f"Available prompts: {[p.name for p in self.mcp_prompts]}")
        except Exception as e:
            logger.error(f"Failed to list prompts: {e}")
            self.mcp_prompts = []

    def get_openai_tools(self): # Convert MCP tools to OpenAI tool format
        openai_tools = []
        for tool in self.mcp_tools:
            if tool.name == "calculator": # Specific handling for known tools
                 openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": tool.description or "Perform mathematical calculations",
                        "parameters": tool.parameters or { # Provide schema if available
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt", "sin", "cos", "tan"]},
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["operation", "a"]
                        }
                    }
                })
            # Add more known tools here
        return openai_tools if openai_tools else None

    async def handle_tool_calls(self, tool_calls: list[ChatCompletionMessageToolCall]):
        """Handle tool calls requested by the LLM."""
        tool_messages = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            tool_call_id = tool_call.id
            try:
                arguments = json.loads(tool_call.function.arguments)
                logger.info(f"Attempting tool call: {function_name}({arguments}) ID: {tool_call_id}")
                
                # Check if it's a known MCP tool
                if any(t.name == function_name for t in self.mcp_tools):
                    tool_result = await self.mcp_session.call_tool(function_name, arguments=arguments)
                    logger.info(f"MCP Tool {function_name} result: {tool_result}")
                    tool_messages.append({ # Message for OpenAI with tool result
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_result), # Result needs to be a string
                    })
                else:
                    logger.warning(f"LLM requested unknown tool: {function_name}")
                    tool_messages.append({ 
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": f"Tool '{function_name}' not found."}),
                    })
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode arguments for tool {function_name}: {tool_call.function.arguments}")
                 tool_messages.append({ 
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": "Invalid arguments format"}),
                 })
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}", exc_info=True)
                tool_messages.append({ 
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps({"error": str(e)}),
                })
        return tool_messages

    async def process_llm_response(self, response_message: ChatCompletionMessage):
        """Process response from OpenAI, checking for tool calls."""
        if response_message.tool_calls:
            logger.info("LLM requested tool calls.")
            # Add the assistant message with tool calls to history
            self.conversation_history.append(response_message.model_dump(exclude_unset=True))
            
            tool_messages = await self.handle_tool_calls(response_message.tool_calls)
            
            # Send tool results back to OpenAI for final response
            messages = self.prepare_messages()
            messages.extend(tool_messages) # Add tool results
            
            logger.info("Sending tool results back to LLM...")
            openai_tools = self.get_openai_tools()
            completion = await self.call_openai(messages, tools=openai_tools) # Don't allow tool calls in the response to tool results
            if completion:
                final_response = completion.choices[0].message
                self.conversation_history.append(final_response.model_dump(exclude_unset=True))
                return final_response.content
            else:
                return "Error getting final response after tool call."
        else:
            # No tool calls, just regular response
            assistant_response = response_message.content
            self.conversation_history.append(response_message.model_dump(exclude_unset=True))
            return assistant_response

    def prepare_messages(self) -> list[dict]:
        """Prepare message list for OpenAI API call."""
        messages = [
            {"role": "system", "content": self.system_context},
        ]
        messages.extend(self.conversation_history)
        return messages

    async def call_openai(self, messages: list[dict], tools=None, tool_choice="auto") -> ChatCompletion | None:
         """Calls the OpenAI API and handles potential errors."""
         if not self.openai_client:
             logger.error("OpenAI client not available.")
             return None
         try:
            logger.debug(f"Sending to OpenAI: Messages={messages}, Tools={tools}")
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            logger.debug(f"Received from OpenAI: {completion.choices[0].message}")
            return completion
         except APIError as e:
             logger.error(f"OpenAI API Error: {e}", exc_info=True)
             print(f"\nError communicating with OpenAI: {e}")
             return None
         except Exception as e:
             logger.error(f"Unexpected error during OpenAI call: {e}", exc_info=True)
             print(f"\nAn unexpected error occurred: {e}")
             return None

    async def handle_prompt_command(self, command_parts: list[str]):
        """Handle the /use_prompt command."""
        if len(command_parts) < 2:
            print("Usage: /use_prompt <prompt_name> [arg_name=value ...]")
            return

        prompt_name = command_parts[1]
        prompt = next((p for p in self.mcp_prompts if p.name == prompt_name), None)

        if not prompt:
            print(f"Error: Prompt '{prompt_name}' not found.")
            print(f"Available prompts: {[p.name for p in self.mcp_prompts]}")
            return
        
        args = {}
        for part in command_parts[2:]:
            if '=' in part:
                key, value = part.split('=', 1)
                args[key] = value
            else:
                print(f"Warning: Ignoring invalid argument format '{part}'. Use key=value.")

        # Validate required arguments (basic check)
        required_args = {arg.name for arg in prompt.arguments if arg.required}
        missing_args = required_args - set(args.keys())
        if missing_args:
            print(f"Error: Missing required arguments for prompt '{prompt_name}': {', '.join(missing_args)}")
            print("Prompt arguments: " + ", ".join([f"{arg.name}{' (req)' if arg.required else ' (opt)'} - {arg.description}" for arg in prompt.arguments]))
            return

        try:
            print(f"\nGenerating prompt '{prompt_name}' with arguments: {args}")
            # Use get_prompt to get the messages from the server based on args
            prompt_result = await self.mcp_session.get_prompt(prompt_name, arguments=args)
            
            # Prepare messages using the prompt result
            prompt_messages = []
            for msg in prompt_result.messages:
                role = "user" if msg.role == SamplingRole.USER else "assistant" if msg.role == SamplingRole.ASSISTANT else "system"
                prompt_messages.append({"role": role, "content": msg.content.text})
            
            print("Sending prompt to LLM...")
            # Call OpenAI with the generated prompt messages
            completion = await self.call_openai(prompt_messages)
            if completion:
                assistant_response = completion.choices[0].message.content
                print(f"\nLLM Response:\n{assistant_response}")
                # Decide whether to add prompt interaction to main history (optional)
                # self.conversation_history.append({"role": "user", "content": f"Used prompt: {prompt_name}({args})"})
                # self.conversation_history.append({"role": "assistant", "content": assistant_response})
            else:
                print("Failed to get response from LLM for the prompt.")
                
        except Exception as e:
            logger.error(f"Error using prompt '{prompt_name}': {e}", exc_info=True)
            print(f"Error executing prompt: {e}")


    async def run_interactive_session(self):
        """Main loop for the interactive chat session."""
        if not self.openai_client: return # Already logged error in init
        
        logger.info("Starting combined client...")
        try:
            async with stdio_client(server_params) as (read, write):
                await self.initialize_mcp(read, write)

                print("\n--- Combined MCP Chatbot --- (type 'quit' to exit)")
                print("Commands: /list_tools, /list_prompts, /use_prompt <name> [args...]")
                print("----------------------------")
                
                while True:
                    user_input = await asyncio.to_thread(input, "\nYou: ")
                    user_input_lower = user_input.lower().strip()

                    if user_input_lower == 'quit':
                        print("Exiting chatbot.")
                        break
                    if not user_input.strip():
                        continue
                        
                    # Handle commands
                    if user_input.startswith('/'):
                        command_parts = user_input.split()
                        command = command_parts[0].lower()
                        if command == '/list_tools':
                            print("\nAvailable Tools:")
                            if self.mcp_tools:
                                for tool in self.mcp_tools:
                                    print(f"- {tool.name}: {tool.description}")
                            else:
                                print("  (No tools available or failed to fetch)")
                        elif command == '/list_prompts':
                            print("\nAvailable Prompts:")
                            if self.mcp_prompts:
                                for prompt in self.mcp_prompts:
                                    arg_parts = []
                                    for arg in prompt.arguments:
                                        part = arg.name
                                        if arg.required:
                                            part += "*"
                                        arg_parts.append(part)
                                    args_desc = ", ".join(arg_parts)
                                    print(f"- {prompt.name}({args_desc}): {prompt.description}")
                            else:
                                print("  (No prompts available or failed to fetch)")
                        elif command == '/use_prompt':
                            await self.handle_prompt_command(command_parts)
                        else:
                            print(f"Unknown command: {command}")
                        continue # Skip LLM call for commands

                    # Regular chat - interact with LLM
                    print("Assistant thinking...")
                    self.conversation_history.append({"role": "user", "content": user_input})
                    
                    messages = self.prepare_messages()
                    openai_tools = self.get_openai_tools()
                    
                    completion = await self.call_openai(messages, tools=openai_tools)
                    
                    if completion:
                        response_message = completion.choices[0].message
                        assistant_output = await self.process_llm_response(response_message)
                        print(f"\nAssistant: {assistant_output}")
                    else:
                         # Error already printed by call_openai
                         self.conversation_history.pop() # Remove user message if call failed
                         
                    # Limit history size
                    if len(self.conversation_history) > 10: # Keep ~5 turns
                        # Find the first user message to keep (preserving system prompt implicitly)
                        first_user_msg_index = -1
                        for i, msg in enumerate(self.conversation_history):
                            if msg["role"] == "user":
                                first_user_msg_index = i
                                break
                        if first_user_msg_index > 0:
                            # Keep messages from the first user message onwards, within the limit
                             keep_from = max(first_user_msg_index, len(self.conversation_history) - 10)
                             self.conversation_history = self.conversation_history[keep_from:]
                         # Simple truncation if only assistant messages somehow
                        elif len(self.conversation_history) > 10:
                             self.conversation_history = self.conversation_history[-10:]
                            
        except ConnectionRefusedError:
            logger.error("Connection refused. Is the server script available and correct?")
            print("\nError: Could not connect to the MCP server.")
        except Exception as e:
            logger.error(f"Client session error: {e}", exc_info=True)
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            if self.mcp_session and self.mcp_session.is_connected:
                logger.info("Closing MCP session.")
                await self.mcp_session.close()

async def main():
    chatbot = CombinedMCPChatbot()
    await chatbot.run_interactive_session()

if __name__ == "__main__":
    # Add rudimentary SIGINT handler for cleaner exit
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nCaught interrupt, shutting down...")
    finally:
        # Cancel lingering tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        # Allow tasks to finish cancellation
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close() 