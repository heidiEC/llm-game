import json
import httpx # For asynchronous HTTP requests
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import asyncio

# Simple stopwords list for demo purposes
STOPWORDS = {"the", "and", "for", "in", "on", "at", "to", "of", "with", "by", "about", "like", "through", "over", "before", "between", "after", "since", "without", "under", "within", "along", "following", "across", "behind", "beyond", "plus", "except", "but", "up", "down", "from", "into", "have", "has", "had", "was", "were", "been", "being", "will", "would", "shall", "should", "may", "might", "can", "could", "must"}

class EverychartMCPClient:
    def __init__(self, server_url, api_key=None):
        self.server_url = server_url.rstrip('/')
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.schema_cache = None
        self.hub_nodes_cache = {} # Simple in-memory cache
        # For a more robust cache, consider libraries like cachetools for TTL, LRU, etc.

    async def get_hub_nodes(self, filter_text: Optional[str] = None, context_hints: Optional[Union[List[str], str]] = None) -> List[Dict[str, Any]]:
        """Get top-level taxonomy categories as hub nodes"""
        cache_key_parts = [str(filter_text)]
        if context_hints:
            if isinstance(context_hints, list):
                cache_key_parts.append(','.join(sorted(context_hints))) # Sort for consistent key
            elif isinstance(context_hints, str):
                cache_key_parts.append(context_hints)
        cache_key = ":".join(cache_key_parts)

        if cache_key in self.hub_nodes_cache:
            return self.hub_nodes_cache[cache_key]

        params = {}
        if filter_text:
            params["filter"] = filter_text
        if context_hints:
            if isinstance(context_hints, list):
                params["context_hints"] = ','.join(context_hints)
            else: # is_string
                params["context_hints"] = context_hints

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/api/hub-nodes",
                params=params,
                headers=self.headers
            )
        response.raise_for_status() # Will raise an exception for 4XX/5XX responses
        hub_nodes = response.json()

        self.hub_nodes_cache[cache_key] = hub_nodes
        # Add a proper cache expiration strategy in a real implementation
        return hub_nodes

    async def get_schema(self) -> Dict[str, Any]:
        """Get the database schema information"""
        if self.schema_cache:
            return self.schema_cache

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/api/schema",
                headers=self.headers
            )
        response.raise_for_status()
        self.schema_cache = response.json()
        return self.schema_cache
    
    async def get_graph(self) -> Dict[str, Any]:
        """Get the entire graph data from the server."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/api/graph",
                headers=self.headers
            )
        response.raise_for_status()
        return response.json()

    async def execute_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the graph database"""
        payload = {"query": cypher, "params": params or {}}
        print(f"DEBUG: Sending Cypher payload to /api/cypher: {payload}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/api/cypher",
                json=payload,
                headers=self.headers
            )
        response.raise_for_status()
        return response.json()

    async def expand_node(self, node_id: str, relationship_types: Optional[Union[List[str], str]] = None, depth: int = 1) -> List[Dict[str, Any]]:
        """Expand a node to get its connected nodes"""
        params = {"depth": depth}
        if relationship_types:
            if isinstance(relationship_types, list):
                params["relationship_types"] = ','.join(relationship_types)
            else: # is_string
                params["relationship_types"] = relationship_types

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/api/expand-node/{node_id}",
                params=params,
                headers=self.headers
            )
        response.raise_for_status()
        return response.json()

    async def get_context_for_query(self, query_text: str, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Get relevant context for an LLM query by:
        1. Identifying key concepts in the query
        2. Finding matching hub nodes
        3. Expanding those nodes to get relevant context
        """
        # Extract potential key concepts (simplified implementation)
        # In a real system, you'd use NLP techniques here (e.g., spaCy, NLTK)
        key_concepts = [word for word in query_text.lower().split()
                        if len(word) > 3 and word not in STOPWORDS]

        hub_nodes = await self.get_hub_nodes(context_hints=key_concepts)

        top_hubs = sorted(hub_nodes, key=lambda x: x.get('relevanceScore', 0), reverse=True)[:3]

        context_nodes = []
        # context_node_ids = set() # To avoid duplicates if expansions overlap

        for hub in top_hubs:
            if len(context_nodes) >= max_nodes:
                break
            
            # Consider fetching fewer nodes if close to max_nodes
            # remaining_capacity = max_nodes - len(context_nodes)
            
            expanded_data = await self.expand_node(hub['id'], depth=2) # 'id' is string from HubNode
            
            # Add nodes ensuring not to exceed max_nodes and avoid duplicates
            for node_data in expanded_data:
                # if node_data['id'] not in context_node_ids: # Assuming 'id' is unique string ID from API
                #    context_nodes.append(node_data)
                #    context_node_ids.add(node_data['id'])
                context_nodes.append(node_data) # Simpler for now, may include duplicates from different expansions
                if len(context_nodes) >= max_nodes:
                    break
            # context_nodes = context_nodes[:max_nodes] # Ensure strict limit after each hub expansion

        return {
            "query": query_text,
            "hub_nodes": top_hubs,
            "context": context_nodes[:max_nodes] # Final trim to max_nodes
        }
    async def execute_json_to_cypher(self, payload: Dict) -> httpx.Response:
        """Send a JSON payload to the server for translation to Cypher and execution."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/api/json-to-cypher",
                json=payload,
                headers=self.headers
            )
        response.raise_for_status()
        return response

    async def subscribe_to_context_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Subscribes to the context stream from the server (Server-Sent Events).
        Yields parsed JSON data from each event.
        """
        url = f"{self.server_url}/api/context-stream"
        timeout = httpx.Timeout(5.0, read=None) # 5s connect, no read timeout for stream
        
        while True: # Outer loop for reconnection
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("GET", url, headers=self.headers) as response:
                        response.raise_for_status() # Check for initial connection errors
                        print(f"Successfully connected to context stream: {url}")
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                data_str = line[len("data:"):].strip()
                                try:
                                    yield json.loads(data_str)
                                except json.JSONDecodeError:
                                    print(f"Warning: Could not decode JSON from SSE data: {data_str}")
                            elif line.startswith("event:"): # Optional: handle custom event types
                                pass # print(f"SSE Event: {line}")
            except httpx.HTTPStatusError as e:
                print(f"HTTP error connecting to stream: {e}. Retrying in 5 seconds...")
            except httpx.RequestError as e:
                print(f"Request error connecting to stream: {e}. Retrying in 5 seconds...")
            except Exception as e:
                print(f"Unexpected error in stream subscription: {e}. Retrying in 5 seconds...")
            
            await asyncio.sleep(5) # Wait before retrying