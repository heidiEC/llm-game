import os
import json
import time
from typing import Dict, List, Optional, Union, Any
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from neo4j import GraphDatabase
import redis
from sse_starlette.sse import EventSourceResponse

# --- Models ---
class NodeData(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class RelationshipData(BaseModel):
    id: str
    type: str
    startNodeId: str
    endNodeId: str
    properties: Dict[str, Any]

class GraphResponse(BaseModel):
    nodes: List[NodeData]
    relationships: List[RelationshipData]

class HubNode(BaseModel):
    id: str
    name: str
    category: str
    description: Optional[str] = None
    nodeCount: int
    relationshipCount: int
    relevanceScore: Optional[float] = None

class SchemaInfo(BaseModel):
    nodeLabels: List[str]
    relationshipTypes: List[str]
    propertyKeys: Dict[str, List[str]]

# --- Neo4j Database ---
class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def get_taxonomy_categories(self, filter_text=None, limit=25):
        query = """
        MATCH (n:TaxonomyCategory)
        WHERE $filter_text IS NULL OR n.name CONTAINS $filter_text
        OPTIONAL MATCH (n)-[r]-(connected)
        WITH n, id(n) as id, labels(n) as labels, count(DISTINCT connected) AS nodeCount, count(r) AS relCount
        RETURN n, id, labels, nodeCount, relCount
        LIMIT $limit
        """
        result = self.execute_query(query, {"filter_text": filter_text, "limit": limit})
        return result

    def get_connected_nodes(self, node_id, relationship_types=None, depth=1):
        relationship_filter = ""
        if relationship_types and len(relationship_types) > 0:
            type_list = '|'.join([f":{rel_type}" for rel_type in relationship_types])
            relationship_filter = f"[r {type_list}]"

        query = f"""
        MATCH (n)-{relationship_filter}*1..{depth}-(connected)
        WHERE id(n) = $node_id
        RETURN connected, id(connected) as id, labels(connected) as labels
        LIMIT 100
        """
        result = self.execute_query(query, {"node_id": node_id})
        return result

    def get_schema_info(self):
        # Get node labels
        labels_query = "CALL db.labels()"
        labels = [record["label"] for record in self.execute_query(labels_query)]

        # Get relationship types
        rel_query = "CALL db.relationshipTypes()"
        rel_types = [record["relationshipType"] for record in self.execute_query(rel_query)]

        # Get property keys for each label
        prop_keys = {}
        for label in labels:
            prop_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) AS key
            RETURN DISTINCT key
            LIMIT 100
            """
            prop_keys[label] = [record["key"] for record in self.execute_query(prop_query)]

        return {
            "nodeLabels": labels,
            "relationshipTypes": rel_types,
            "propertyKeys": prop_keys
        }

    def execute_cypher(self, query, parameters=None):
        try:
            return self.execute_query(query, parameters)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")

# --- Everychart MCP Server ---
class EverychartMCPServer:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, redis_url=None):
        self.db = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password)
        self.app = FastAPI(title="Everychart MCP Server")

        # Set up Redis for caching if URL provided
        self.redis = None
        if redis_url:
            self.redis = redis.from_url(redis_url)

        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.get("/api/hub-nodes", response_model=List[HubNode])
        async def get_hub_nodes(
            filter: Optional[str] = None,
            context_hints: Optional[str] = None
        ):
            # Get taxonomy categories with counts
            hub_nodes_data = self.db.get_taxonomy_categories(filter_text=filter)

            # Process into HubNode format
            hub_nodes = []
            for data in hub_nodes_data:
                node = data['n']
                hub_node = HubNode(
                    id=str(data['id']),
                    name=node.get('name', f"Category {data['id']}"),
                    category=data['labels'][0] if data['labels'] else "Unknown",
                    description=node.get('description', None),
                    nodeCount=data['nodeCount'],
                    relationshipCount=data['relCount']
                )
                hub_nodes.append(hub_node)

            # Calculate relevance scores if context hints provided
            if context_hints:
                hints = context_hints.split(',')
                for hub_node in hub_nodes:
                    # Simple relevance scoring based on text matching
                    score = 0
                    for hint in hints:
                        if hint.lower() in hub_node.name.lower():
                            score += 1
                        if hub_node.description and hint.lower() in hub_node.description.lower():
                            score += 0.5
                    hub_node.relevanceScore = score

                # Sort by relevance score
                hub_nodes.sort(key=lambda x: x.relevanceScore or 0, reverse=True)

            return hub_nodes

        @self.app.get("/api/schema", response_model=SchemaInfo)
        async def get_schema():
            # Use Redis cache if available
            if self.redis:
                cached = self.redis.get("schema_info")
                if cached:
                    return json.loads(cached)

            schema_info = self.db.get_schema_info()

            # Cache for 1 hour if Redis available
            if self.redis:
                self.redis.setex("schema_info", 3600, json.dumps(schema_info))

            return schema_info

        @self.app.post("/api/cypher")
        async def execute_cypher(query: str, params: Optional[Dict] = None):
            return self.db.execute_cypher(query, params)

        @self.app.get("/api/expand-node/{node_id}")
        async def expand_node(
            node_id: str,
            relationship_types: Optional[str] = None,
            depth: int = 1
        ):
            rel_types = relationship_types.split(',') if relationship_types else None
            expanded = self.db.get_connected_nodes(node_id, rel_types, depth)
            return expanded

        async def context_event_generator():
            # In a real scenario, this would listen for events that trigger context updates
            # For now, we'll just send a simple message every few seconds
            import asyncio # Required for await asyncio.sleep
            i = 0
            while True:
                yield {"event": "update", "data": f"Context update {i}"}
                i += 1
                await asyncio.sleep(5)

        @self.app.get("/api/context-stream")
        async def context_stream():
            return EventSourceResponse(context_event_generator())

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

    def shutdown(self):
        self.db.close()

# Example usage
if __name__ == "__main__":
    # Set up and run the server
    server = EverychartMCPServer(
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379"
    )

    # Run in a separate thread/process in a real application
    import threading
    def run_server():
        server.run(port=8000)
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    time.sleep(1) # Give the server a moment to start

    # Client usage example (would be in a separate process)
    import asyncio
    from mcp_client import EverychartMCPClient # Import client from its new file

    async def client_example():
        client = EverychartMCPClient("http://localhost:8000")

        # Get hub nodes
        hub_nodes = await client.get_hub_nodes(context_hints=["technology", "AI"])
        print(f"Found {len(hub_nodes)} hub nodes")

        # Get schema
        schema = await client.get_schema()
        print(f"Schema has {len(schema['nodeLabels'])} node labels")

        # Execute a query
        results = await client.execute_query(
            "MATCH (n:TechnicalConcept) RETURN n LIMIT 5"
        )
        print(f"Query returned {len(results)} results")

        # Get context for an LLM query
        context = await client.get_context_for_query(
            "How has machine learning impacted healthcare?"
        )
        print(f"Context includes {len(context['context'])} nodes")

    asyncio.run(client_example())

    # Give time to finish and then shutdown
    time.sleep(2)
    server.shutdown()
    server_thread.join()
