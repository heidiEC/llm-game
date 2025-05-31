import os
import json
import time
import re
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
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = []
                for record in result:
                    try:
                        if hasattr(record, 'data'):
                            records.append(record.data())
                        else:
                            # Fallback: convert to dict manually
                            record_dict = {}
                            for key in record.keys():
                                record_dict[key] = record[key]
                            records.append(record_dict)
                    except Exception as e:
                        print(f"WARNING: Could not convert record in execute_query: {e}")
                        continue
                return records
        except Exception as e:
            print(f"ERROR in execute_query: {e}")
            raise

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
        try:
            # Convert node_id to integer if it's a string that looks like a number
            if isinstance(node_id, str) and node_id.isdigit():
                node_id = int(node_id)
        
            relationship_filter = ""
            if relationship_types and len(relationship_types) > 0:
                # Clean relationship types
                clean_types = [rel_type.strip() for rel_type in relationship_types if rel_type.strip()]
                if clean_types:
                    type_list = '|'.join([f":{rel_type}" for rel_type in clean_types])
                    relationship_filter = f"[r{type_list}]"
        
            # Use elementId() for Neo4j 5.x compatibility
            query = f"""
            MATCH (n)-{relationship_filter}*1..{depth}-(connected)
            WHERE elementId(n) = $node_id OR id(n) = $node_id
            RETURN DISTINCT connected, elementId(connected) as elementId, id(connected) as id, labels(connected) as labels
            LIMIT 100
            """
        
            print(f"DEBUG: Executing get_connected_nodes query: {query} with node_id: {node_id}")
            result = self.execute_query(query, {"node_id": node_id})
            print(f"DEBUG: get_connected_nodes returned {len(result)} results")
            return result
        
        except Exception as e:
            print(f"ERROR in get_connected_nodes: {e}")
            # Return empty result instead of failing
            return []

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
                results = []
                # Split multiple statements if they exist
                statements = [stmt.strip() for stmt in query.strip().split(';') if stmt.strip()]
                
                for statement in statements:
                    print(f"DEBUG: Executing statement: {statement}")
                    try:
                        result = session.run(statement, parameters or {})
                        
                        # Check if this is a write operation
                        if any(keyword in statement.upper() for keyword in ["CREATE", "MERGE", "SET", "DELETE", "REMOVE"]):
                            # For write operations, get the summary
                            summary = result.consume()
                            if summary and hasattr(summary, 'counters') and summary.counters:
                                # Convert counters to dict safely
                                counters_dict = {}
                                if hasattr(summary.counters, '_asdict'):
                                    counters_dict = summary.counters._asdict()
                                else:
                                    # Fallback: manually extract counter values
                                    counters_dict = {
                                        'nodes_created': getattr(summary.counters, 'nodes_created', 0),
                                        'nodes_deleted': getattr(summary.counters, 'nodes_deleted', 0),
                                        'relationships_created': getattr(summary.counters, 'relationships_created', 0),
                                        'relationships_deleted': getattr(summary.counters, 'relationships_deleted', 0),
                                        'properties_set': getattr(summary.counters, 'properties_set', 0),
                                        'labels_added': getattr(summary.counters, 'labels_added', 0),
                                        'labels_removed': getattr(summary.counters, 'labels_removed', 0),
                                        'indexes_added': getattr(summary.counters, 'indexes_added', 0),
                                        'indexes_removed': getattr(summary.counters, 'indexes_removed', 0),
                                        'constraints_added': getattr(summary.counters, 'constraints_added', 0),
                                        'constraints_removed': getattr(summary.counters, 'constraints_removed', 0),
                                    }
                                
                                result_data = {
                                    "counters": counters_dict,
                                    "notifications": [str(n) for n in (summary.notifications or [])]
                                }
                            else:
                                result_data = {
                                    "counters": {},
                                    "notifications": []
                                }
                            results.append(result_data)
                        else:
                            # For read operations, get the records
                            records = []
                            for record in result:
                                # Convert record to dict safely
                                try:
                                    if hasattr(record, 'data'):
                                        records.append(record.data())
                                    elif hasattr(record, '_asdict'):
                                        records.append(record._asdict())
                                    else:
                                        # Fallback: convert to dict manually
                                        record_dict = {}
                                        for key in record.keys():
                                            record_dict[key] = record[key]
                                        records.append(record_dict)
                                except Exception as e:
                                    print(f"WARNING: Could not convert record to dict: {e}")
                                    records.append({"error": f"Could not convert record: {str(e)}"})
                            results.append(records)
                            
                    except Exception as e:
                        print(f"ERROR: Failed to execute statement '{statement}': {e}")
                        results.append({"error": str(e), "statement": statement})
                        
                return results
                
        except Exception as e:
            print(f"ERROR executing Cypher: {query} with params {parameters}. Error: {e}")
            import traceback
            traceback.print_exc()
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

                print(f"DEBUG: Processing payload with {len(nodes)} nodes and {len(relationships)} relationships")

                # Create nodes first (keeping existing logic)
                for i, node_data in enumerate(nodes):
                    try:
                        name = node_data.get("name")
                        labels = node_data.get("labels", [])
                        properties = node_data.get("properties", {})
                        
                        if not name:
                            print(f"WARNING: Node {i} has no name, skipping")
                            continue
                        
                        print(f"DEBUG: Processing node '{name}' with labels {labels}")
                        
                        # Clean labels - remove spaces and special characters
                        clean_labels = []
                        for label in labels:
                            clean_label = re.sub(r'[^a-zA-Z0-9_]', '', label.replace(' ', ''))
                            if clean_label:
                                clean_labels.append(clean_label)
                        
                        label_str = ":" + ":".join(clean_labels) if clean_labels else ""
                        
                        # Build properties string more carefully
                        all_properties = {"name": name}
                        all_properties.update(properties)
                        
                        props_list = []
                        for k, v in all_properties.items():
                            # Clean property key
                            clean_key = re.sub(r'[^a-zA-Z0-9_]', '', k.replace(' ', '_'))
                            if not clean_key:
                                continue
                            
                            if isinstance(v, str):
                                # Escape quotes and handle multiline strings
                                escaped_v = v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
                                # Truncate very long strings
                                if len(escaped_v) > 1000:
                                    escaped_v = escaped_v[:1000] + "..."
                                props_list.append(f"{clean_key}: '{escaped_v}'")
                            elif isinstance(v, (int, float)):
                                props_list.append(f"{clean_key}: {v}")
                            elif isinstance(v, bool):
                                props_list.append(f"{clean_key}: {str(v).lower()}")
                            elif v is None:
                                props_list.append(f"{clean_key}: null")
                            else:
                                # For complex objects, convert to JSON string
                                try:
                                    json_str = json.dumps(v)
                                    escaped_v = json_str.replace("\\", "\\\\").replace("'", "\\'")
                                    props_list.append(f"{clean_key}: '{escaped_v}'")
                                except:
                                    # If JSON serialization fails, convert to string
                                    str_v = str(v).replace("\\", "\\\\").replace("'", "\\'")
                                    props_list.append(f"{clean_key}: '{str_v}'")
                        
                        if props_list:
                            props_str = ", ".join(props_list)
                            cypher_command = f"MERGE (n{label_str} {{{props_str}}})"
                            cypher_commands.append(cypher_command)
                            print(f"DEBUG: Generated node Cypher: {cypher_command}")
                        
                    except Exception as e:
                        print(f"ERROR: Failed to process node {i}: {e}")
                        continue

                # Execute node creation first
                if cypher_commands:
                    print(f"DEBUG: Executing {len(cypher_commands)} node creation commands first")
                    try:
                        combined_node_cypher = ";\n".join(cypher_commands) + ";"
                        node_results = self.db.execute_cypher(combined_node_cypher)
                        print(f"DEBUG: Node creation results: {node_results}")
                    except Exception as e:
                        print(f"ERROR: Failed to create nodes: {e}")
                        return {"success": False, "message": f"Node creation failed: {str(e)}"}

                # Now create relationships separately
                relationship_commands = []
                for i, rel_data in enumerate(relationships):
                    try:
                        source_name = rel_data.get("source_node_name")
                        target_name = rel_data.get("target_node_name")
                        rel_type = rel_data.get("type")
                        properties = rel_data.get("properties", {})
                        
                        if not all([source_name, target_name, rel_type]):
                            print(f"WARNING: Relationship {i} missing required fields, skipping")
                            continue
                        
                        print(f"DEBUG: Processing relationship {source_name} -> {target_name} ({rel_type})")
                        
                        # First, let's check if both nodes exist
                        escaped_source = source_name.replace("\\", "\\\\").replace("'", "\\'")
                        escaped_target = target_name.replace("\\", "\\\\").replace("'", "\\'")
                        
                        # Check if source node exists
                        check_source_query = f"MATCH (n {{name: '{escaped_source}'}}) RETURN count(n) as count"
                        try:
                            source_check = self.db.execute_query(check_source_query)
                            source_count = source_check[0]['count'] if source_check else 0
                            print(f"DEBUG: Source node '{source_name}' count: {source_count}")
                        except Exception as e:
                            print(f"ERROR: Could not check source node: {e}")
                            continue
                        
                        # Check if target node exists
                        check_target_query = f"MATCH (n {{name: '{escaped_target}'}}) RETURN count(n) as count"
                        try:
                            target_check = self.db.execute_query(check_target_query)
                            target_count = target_check[0]['count'] if target_check else 0
                            print(f"DEBUG: Target node '{target_name}' count: {target_count}")
                        except Exception as e:
                            print(f"ERROR: Could not check target node: {e}")
                            continue
                        
                        if source_count == 0:
                            print(f"WARNING: Source node '{source_name}' not found, skipping relationship")
                            continue
                        
                        if target_count == 0:
                            print(f"WARNING: Target node '{target_name}' not found, skipping relationship")
                            continue
                        
                        # Clean relationship type
                        clean_rel_type = re.sub(r'[^a-zA-Z0-9_]', '', rel_type.replace(' ', '_').upper())
                        
                        props_str = ""
                        if properties:
                            prop_pairs = []
                            for k, v in properties.items():
                                clean_key = re.sub(r'[^a-zA-Z0-9_]', '', k.replace(' ', '_'))
                                if not clean_key:
                                    continue
                                    
                                if isinstance(v, str):
                                    escaped_v = v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
                                    if len(escaped_v) > 1000:
                                        escaped_v = escaped_v[:1000] + "..."
                                    prop_pairs.append(f"{clean_key}: '{escaped_v}'")
                                elif isinstance(v, (int, float)):
                                    prop_pairs.append(f"{clean_key}: {v}")
                                elif isinstance(v, bool):
                                    prop_pairs.append(f"{clean_key}: {str(v).lower()}")
                                elif v is None:
                                    prop_pairs.append(f"{clean_key}: null")
                                else:
                                    try:
                                        json_str = json.dumps(v)
                                        escaped_v = json_str.replace("\\", "\\\\").replace("'", "\\'")
                                        prop_pairs.append(f"{clean_key}: '{escaped_v}'")
                                    except:
                                        str_v = str(v).replace("\\", "\\\\").replace("'", "\\'")
                                        prop_pairs.append(f"{clean_key}: '{str_v}'")
                            
                            if prop_pairs:
                                props_str = " {" + ", ".join(prop_pairs) + "}"
                        
                        cypher_command = f"MATCH (source {{name: '{escaped_source}'}}), (target {{name: '{escaped_target}'}}) MERGE (source)-[:{clean_rel_type}{props_str}]->(target)"
                        relationship_commands.append(cypher_command)
                        print(f"DEBUG: Generated relationship Cypher: {cypher_command}")
                        
                    except Exception as e:
                        print(f"ERROR: Failed to process relationship {i}: {e}")
                        continue

                # Execute relationship creation
                relationship_results = []
                if relationship_commands:
                    print(f"DEBUG: Executing {len(relationship_commands)} relationship creation commands")
                    try:
                        combined_rel_cypher = ";\n".join(relationship_commands) + ";"
                        relationship_results = self.db.execute_cypher(combined_rel_cypher)
                        print(f"DEBUG: Relationship creation results: {relationship_results}")
                    except Exception as e:
                        print(f"ERROR: Failed to create relationships: {e}")
                        return {"success": False, "message": f"Relationship creation failed: {str(e)}"}

                # Return combined results
                total_commands = len(cypher_commands) + len(relationship_commands)
                if total_commands > 0:
                    return {
                        "success": True, 
                        "message": f"Graph updated: {len(cypher_commands)} nodes, {len(relationship_commands)} relationships", 
                        "node_results": node_results if cypher_commands else [],
                        "relationship_results": relationship_results,
                        "commands_executed": total_commands
                    }
                else:
                    return {"success": True, "message": "No valid nodes or relationships to add"}

            except HTTPException as e:
                raise e
            except Exception as e:
                print(f"ERROR: Exception in json_to_cypher: {e}")
                import traceback
                traceback.print_exc()
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
            try:
                print(f"DEBUG: expand_node called with node_id={node_id}, relationship_types={relationship_types}, depth={depth}")
        
                rel_types = relationship_types.split(',') if relationship_types else None
                expanded = self.db.get_connected_nodes(node_id, rel_types, depth)
        
                print(f"DEBUG: expand_node returning {len(expanded)} results")
                return {"success": True, "data": expanded}
        
            except Exception as e:
                print(f"ERROR in expand_node endpoint: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": str(e), "data": []}

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

        # Run the server
        print("INFO: Starting MCP server on port 8000...")
        server.run(port=8000)
        
        # Remove the client example and threading code since it's causing issues
        # The server will run normally and the game client will connect to it
