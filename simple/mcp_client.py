#!/usr/bin/env python3
import os
import logging
import asyncio
import sys
from mcp import (
    ClientSession, 
    StdioServerParameters, 
    # Import specific types needed
    Resource, 
    ListResourcesRequest # For listing resources if needed
)
from mcp.client.stdio import stdio_client # Import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define server parameters for stdio connection
server_params = StdioServerParameters(
    command=sys.executable,  # Use the current Python interpreter
    args=["mcp_server.py"], # Argument is the server script
    # env=None,  # Optional environment variables
)

class MCPHostApp:
    """
    MCP host application connecting via stdio and using OpenAI.
    """
    def __init__(self):
        # OpenAI client is still needed here
        try:
            self.openai_client = OpenAI()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    async def run_chat(self):
        """Connect to server, run interactive chat."""
        if not self.openai_client:
             print("OpenAI client not initialized. Check API key. Exiting.")
             return
             
        logger.info("Starting stdio client...")
        try:
            async with stdio_client(server_params) as (read, write):
                logger.info("stdio transport connected, creating client session.")
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session...")
                    server_info = await session.initialize()
                    logger.info(f"Connected to server: {server_info.server_name}")
                    
                    print("\nSimple MCP Chat (type 'quit' to exit)")
                    print("--------------------------------------")
                    
                    conversation_history = []
                    
                    while True:
                        user_input = await asyncio.to_thread(input, "\nYou: ") # Use async input
                        if user_input.lower() in ['quit', 'exit']:
                            print("Exiting chat.")
                            break
                            
                        if not user_input.strip():
                            continue
                        
                        print("Generating response...")
                        # Get context from MCP server
                        system_context, mcp_info_context = await self.get_mcp_context(session)
                        
                        # Prepare messages for OpenAI
                        messages = []
                        if system_context:
                            messages.append({"role": "system", "content": system_context})
                        else:
                            # Fallback if context not found
                            messages.append({"role": "system", "content": "You are a helpful assistant."})
                        
                        # Optionally add other context like mcp_info
                        # if mcp_info_context:
                        #     messages.append({"role": "system", "content": f"Context: {mcp_info_context}"})
                        
                        # Add conversation history
                        messages.extend(conversation_history)
                        # Add current user message
                        messages.append({"role": "user", "content": user_input})
                        
                        # Call OpenAI
                        try:
                            response = self.openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages
                            )
                            assistant_response = response.choices[0].message.content
                        except Exception as e:
                            logger.error(f"OpenAI API error: {e}")
                            assistant_response = f"Error generating response: {str(e)}"

                        print(f"\nAssistant: {assistant_response}")
                        
                        # Update history
                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": assistant_response})
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]
                            
        except ConnectionRefusedError:
            logger.error("Connection refused. Is the server script available and correct?")
        except Exception as e:
            logger.error(f"Client session error: {e}", exc_info=True)

    async def get_mcp_context(self, session: ClientSession):
        """Retrieve context resources from the MCP server."""
        system_context = None
        mcp_info_context = None
        try:
            logger.info("Reading system context resource (context://system)")
            content, mime_type = await session.read_resource("context://system")
            # Assuming content is string or bytes, decode if necessary
            if isinstance(content, bytes):
                 system_context = content.decode('utf-8')
            else:
                 system_context = str(content)
            logger.info(f"Got system context (type: {mime_type})")
        except Exception as e:
            logger.warning(f"Could not read system context: {e}")
            
        try:
            logger.info("Reading MCP info resource (info://mcp)")
            content, mime_type = await session.read_resource("info://mcp")
            if isinstance(content, bytes):
                 mcp_info_context = content.decode('utf-8')
            else:
                 mcp_info_context = str(content)
            logger.info(f"Got MCP info (type: {mime_type})")
        except Exception as e:
            logger.warning(f"Could not read MCP info: {e}")
            
        return system_context, mcp_info_context

async def main():
    host_app = MCPHostApp()
    await host_app.run_chat()

if __name__ == "__main__":
    asyncio.run(main()) 