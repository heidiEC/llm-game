import os
import json
import time
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from neo4j import GraphDatabase
import redis
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
load_dotenv()

import certifi

# --- Attempt to set SSL_CERT_FILE programmatically using certifi ---
try:
    certifi_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi_path
    os.environ['REQUESTS_CA_BUNDLE'] = certifi_path # For other libs like requests
    print(f"INFO: Programmatically set SSL_CERT_FILE to: {certifi_path}")
except ImportError:
    print("WARNING: certifi package not found. Cannot programmatically set SSL_CERT_FILE.")
except Exception as e:
    print(f"ERROR: Error setting SSL_CERT_FILE from certifi: {e}")
# --- End of SSL_CERT_FILE setting ---

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
    propertyKeys: Dict[str, List[str]] # Existing
    # Optional: Add more detailed APOC-like schema structure if desired
    # apocSchema: Optional[List[Dict[str, Any]]] = None 


# --- Neo4j Database ---
class Neo4jDatabase:
    def __init__(self, uri, user, password):
        print(f"Connecting to Neo4j with URI: {uri}, User: {user}")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            # For write queries, we might want to return summary info
            # For read queries, we return records
            # This simple version just returns records for now, see execute_cypher for more detail
            tx = session.begin_transaction()
            result = tx.run(query, parameters or {})
            records = [record.data() for record in result]
            tx.commit()
            return records

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
        # Attempt to use APOC for richer schema, fallback to basic if APOC is not available or fails
        apoc_schema_query = """
        CALL apoc.meta.data() YIELD label, property, type, other, unique, index, elementType
        WHERE elementType = 'node' AND NOT label STARTS WITH '_'
        WITH label,
            collect(CASE WHEN type <> 'RELATIONSHIP' THEN [property, type + CASE WHEN unique THEN " unique" ELSE "" END + CASE WHEN index THEN " indexed" ELSE "" END] END) AS attributes,
            collect(CASE WHEN type = 'RELATIONSHIP' THEN [property, head(other)] END) AS relationships
        RETURN label, apoc.map.fromPairs(attributes) AS attributes, apoc.map.fromPairs(relationships) AS relationships
        """
        try:
            apoc_results = self.execute_query(apoc_schema_query)
            # Process apoc_results into your desired SchemaInfo structure
            # This might involve extracting distinct labels, rel types, and properties per label
            node_labels = sorted(list(set([item['label'] for item in apoc_results])))
            
            all_rel_types = set()
            property_keys_map = {label: [] for label in node_labels}

            for item in apoc_results:
                label = item['label']
                if item.get('attributes'):
                    property_keys_map[label].extend(item['attributes'].keys())
                if item.get('relationships'):
                    all_rel_types.update(item['relationships'].keys())
            
            # Deduplicate property keys
            for label in property_keys_map:
                property_keys_map[label] = sorted(list(set(property_keys_map[label])))

            return {
                "nodeLabels": node_labels,
                "relationshipTypes": sorted(list(all_rel_types)),
                "propertyKeys": property_keys_map
                # "apocSchema": apoc_results # Optionally include the raw APOC output
            }
        except Exception as e:
            print(f"WARNING: APOC schema query failed ('{e}'). Falling back to basic schema introspection. Ensure APOC is installed.")
            # Fallback to basic schema introspection
            labels = [record["label"] for record in self.execute_query("CALL db.labels()")]
            rel_types = [record["relationshipType"] for record in self.execute_query("CALL db.relationshipTypes()")]
            prop_keys = {}
            for label in labels:
                prop_query = f"MATCH (n:{label}) UNWIND keys(n) AS key RETURN DISTINCT key LIMIT 100"
                prop_keys[label] = [record["key"] for record in self.execute_query(prop_query)]
            return {
                "nodeLabels": labels,
                "relationshipTypes": rel_types,
                "propertyKeys": prop_keys
            }

    def execute_cypher(self, query: str, parameters: Optional[Dict] = None):
        try:
            with self.driver.session() as session:
                tx = session.begin_transaction()
                results = []
                for statement in query.strip().split(';'):
                    if statement.strip():  # Ignore empty statements
                        result = tx.run(statement, parameters or {})
                        if any(keyword in statement.upper() for keyword in ["CREATE", "MERGE", "SET", "DELETE", "REMOVE"]):
                            summary = result.consume()  # This returns ResultSummary
                            if summary and summary.counters:
                                # Access counters directly as properties, not ._counts
                                counters_dict = {
                                    "nodes_created": summary.counters.nodes_created,
                                    "nodes_deleted": summary.counters.nodes_deleted,
                                    "relationships_created": summary.counters.relationships_created,
                                    "relationships_deleted": summary.counters.relationships_deleted,
                                    "properties_set": summary.counters.properties_set,
                                    "labels_added": summary.counters.labels_added,
                                    "labels_removed": summary.counters.labels_removed,
                                    "indexes_added": summary.counters.indexes_added,
                                    "indexes_removed": summary.counters.indexes_removed,
                                    "constraints_added": summary.counters.constraints_added,
                                    "constraints_removed": summary.counters.constraints_removed,
                                }
                                results.append({
                                    "counters": counters_dict, 
                                    "notifications": [notif._asdict() for notif in (summary.notifications or [])]
                                })
                            else:
                                results.append({
                                    "notifications": [notif._asdict() for notif in (summary.notifications or [])] if summary else []
                                })
                        else:
                            # For read queries, return the actual data
                            results.append([record.data() for record in result])
                tx.commit()
                return results
        except Exception as e:
            print(f"ERROR executing Cypher: {query} with params {parameters}. Error: {e}")
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
        async def run_cypher_query(payload: Dict): # Changed to accept a JSON payload
            query = payload.get("query")
            params = payload.get("params")
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query' in request payload")
            # The db.execute_cypher method now handles potential errors and returns structured info
            return self.db.execute_cypher(query, params or {})
        

        @self.app.post("/api/json-to-cypher")
        async def json_to_cypher(payload: Dict):
            cypher_commands = []
            try:
                explanation = payload.get("explanation", "")
                nodes = payload.get("nodes", [])
                relationships = payload.get("relationships", [])
                future_directions = payload.get("future_directions", [])

                # Create nodes
                for node_data in nodes:
                    name = node_data.get("name")
                    labels = node_data.get("labels", [])
                    properties = node_data.get("properties", {})
                    if name:
                        # Escape single quotes in name and properties
                        escaped_name = name.replace("'", "\\'")
                        label_str = ":" + ":".join(labels) if labels else ""
                        
                        # Build properties string more carefully
                        prop_parts = [f"name: '{escaped_name}'"]
                        for k, v in properties.items():
                            if isinstance(v, str):
                                escaped_v = v.replace("'", "\\'")
                                prop_parts.append(f"{k}: '{escaped_v}'")
                            elif isinstance(v, (int, float, bool)):
                                prop_parts.append(f"{k}: {json.dumps(v)}")
                            else:
                                prop_parts.append(f"{k}: {json.dumps(v)}")
                        
                        props_str = ", ".join(prop_parts)
                        cypher_commands.append(f"MERGE (n{label_str} {{{props_str}}})")

                # Create relationships
                for rel_data in relationships:
                    source_name = rel_data.get("source_node_name")
                    target_name = rel_data.get("target_node_name")
                    rel_type = rel_data.get("type")
                    properties = rel_data.get("properties", {})
                    
                    if source_name and target_name and rel_type:
                        # Escape single quotes
                        escaped_source = source_name.replace("'", "\\'")
                        escaped_target = target_name.replace("'", "\\'")
                        
                        props_str = ""
                        if properties:
                            prop_parts = []
                            for k, v in properties.items():
                                if isinstance(v, str):
                                    escaped_v = v.replace("'", "\\'")
                                    prop_parts.append(f"{k}: '{escaped_v}'")
                                elif isinstance(v, (int, float, bool)):
                                    prop_parts.append(f"{k}: {json.dumps(v)}")
                                else:
                                    prop_parts.append(f"{k}: {json.dumps(v)}")
                            props_str = " {" + ", ".join(prop_parts) + "}"
                        
                        cypher_commands.append(
                            f"MATCH (source {{name: '{escaped_source}'}}), (target {{name: '{escaped_target}'}}) "
                            f"MERGE (source)-[:{rel_type}{props_str}]->(target)"
                        )

                # Execute the Cypher commands
                if cypher_commands:
                    combined_cypher = ";\n".join(cypher_commands)
                    print(f"DEBUG: Executing combined Cypher:\n{combined_cypher}")
                    result = self.db.execute_cypher(combined_cypher)
                    return {"success": True, "message": "Graph updated", "cypher_result": result}
                else:
                    return {"success": True, "message": "No nodes or relationships to add"}

            except HTTPException as e:
                raise e
            except Exception as e:
                print(f"Error processing JSON to Cypher: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")

        @self.app.get("/api/graph", response_model=GraphResponse)
        async def get_graph():
            nodes_data = self.db.execute_query("MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties")
            relationships_data = self.db.execute_query("MATCH ()-[r]->() RETURN elementId(r) AS id, type(r) AS type, elementId(startNode(r)) AS startNodeId, elementId(endNode(r)) AS endNodeId, properties(r) AS properties")

            nodes = [
                NodeData(
                    id=item['id'],
                    labels=item['labels'],
                    properties=item['properties']
                )
                for item in nodes_data
            ]
            relationships = [
                RelationshipData(
                    id=item['id'],
                    type=item['type'],
                    startNodeId=item['startNodeId'],
                    endNodeId=item['endNodeId'],
                    properties=item['properties']
                )
                for item in relationships_data
            ]

            return GraphResponse(nodes=nodes, relationships=relationships)

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
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    redis_url = os.environ.get("REDIS_URL") 

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Error: Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
    else:
        server = EverychartMCPServer(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            redis_url=redis_url
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
