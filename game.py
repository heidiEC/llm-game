import os
import json
import time
from typing import Dict, List, Optional, Union, Any
import asyncio
from neo4j import GraphDatabase
from anthropic import Anthropic
import ollama
import hashlib
import httpx
from mcp_client import EverychartMCPClient  # Import the client
from dotenv import load_dotenv
load_dotenv()

# --- Environment Variables (Make sure these are set) ---
#NEO4J_URI = os.environ.get("NEO4J_URI")
#NEO4J_USER = os.environ.get("NEO4J_USER")
#NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")

class KnowledgeGraphGame:
    def __init__(self):
        # Connect to Neo4j (still used for initial setup and potentially direct access)
        #self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Initialize Anthropic client
        if ANTHROPIC_API_KEY:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self.client = None
            print("WARNING: ANTHROPIC_API_KEY not set.")

        # Initialize Ollama client
        try:
            self.ollama_client = ollama.Client()
            self.ollama_client.list()
            self.ollama_available = True
            print("INFO: Ollama client initialized.")
        except Exception as e:
            self.ollama_client = None
            self.ollama_available = False
            print(f"WARNING: Could not initialize Ollama client: {e}")

        # Initialize MCP Client
        self.mcp_client = EverychartMCPClient(MCP_SERVER_URL)

        # Initialize game state
        self.turn_count = 0
        self.participants = []
        self.current_graph_state = {"nodes": {}, "relationships": {}}

        self.load_participants()
        asyncio.run(self.initialize_graph())
        asyncio.run(self.update_current_graph_state_from_mcp())

    def load_participants(self):
        print("INFO: Participants managed in-memory.")
        pass

    def save_participants(self):
        pass

    async def update_current_graph_state_from_mcp(self):
        """Fetch the current graph state from the MCP server."""
        graph_data = await self.mcp_client.get_graph()
        nodes = {}
        for node in graph_data.get("nodes", []):
            nodes[node['id']] = {"properties": node['properties'], "labels": node['labels'], "id": node['id']}

        relationships = {}
        for rel in graph_data.get("relationships", []):
            relationships[(rel['startNodeId'], rel['endNodeId'], rel['type'])] = rel['properties']

        self.current_graph_state = {
            "nodes": nodes,
            "relationships": relationships
        }

    async def fetch_graph_state_from_neo4j(self):
        """Fallback to fetch graph state directly from Neo4j."""
        print("WARNING: Falling back to fetch graph state directly from Neo4j.")
        with self.driver.session() as session:
            nodes = {}
            nodes_result = session.run("MATCH (n) RETURN elementId(n) AS id, n, labels(n) AS labels")
            for record in nodes_result:
                nodes[record["id"]] = {"properties": dict(record["n"]), "labels": record["labels"], "id": record["id"]}

            relationships = {}
            rels_result = session.run("MATCH ()-[r]->() RETURN elementId(r) AS id, r, elementId(startNode(r)) AS source, elementId(endNode(r)) AS target, type(r) AS type")
            for record in rels_result:
                relationships[record["id"]] = {"properties": dict(record["r"]), "source": record["source"], "target": record["target"], "type": record["type"], "id": record["id"]}

            self.current_graph_state = {"nodes": nodes, "relationships": relationships}
            print("INFO: Graph state fetched directly from Neo4j.")

    async def initialize_graph(self, force_reset=False):
        """Ensure the foundational knowledge graph nodes and relationships exist via MCP."""
        if force_reset:
            print("INFO: Force reset requested. Clearing entire graph via MCP...")
            try:
                response = await self.mcp_client.execute_query("MATCH (n) DETACH DELETE n")
                print("INFO: Graph cleared via MCP.")
            except Exception as e:
                print(f"ERROR: Could not clear graph via MCP: {e}")

        print("INFO: Ensuring foundational seed nodes and relationships exist via MCP...")
        seed_cypher = """
        MERGE (biomimeticAlgo:Concept:ComputerScience {name: "Biomimetic Algorithms", description: "Algorithms inspired by biological processes in nature", domain: "Computer Science"});
        MERGE (antColony:Concept:Biology {name: "Ant Colony Optimization", description: "Swarm intelligence based on pheromone trail optimization", domain: "Biology"});
        MERGE (networkRouting:Concept:Telecommunications {name: "Network Routing Algorithms", description: "Methods for determining optimal paths in communication networks", domain: "Telecommunications"});
        MERGE (collectiveIntelligence:Concept:Behavior {name: "Collective Intelligence", description: "Shared or group intelligence emerging from collaboration", domain: "Complex Systems"});
        MERGE (biomimeticAlgo)-[:INSPIRED_BY {analysis: "Biomimetic algorithms take inspiration from the principles of collective intelligence found in biological systems."}]->(collectiveIntelligence);
        MERGE (antColony)-[:DEMONSTRATES {analysis: "Ant colony optimization is a specific example of a system that demonstrates collective intelligence."}]->(collectiveIntelligence);
        MERGE (biomimeticAlgo)-[:APPLIED_TO {analysis: "Biomimetic algorithms, such as ant colony optimization, are applied to solve problems like network routing."}]->(networkRouting);
        MERGE (antColony)-[:SERVES_AS_MODEL_FOR {analysis: "The behavior of ant colonies serves as a model for the design of biomimetic algorithms like ACO."}]->(biomimeticAlgo)
        """
        try:
            await self.mcp_client.execute_query(seed_cypher)
            print("INFO: Seed nodes and relationships ensured via MCP.")
        except Exception as e:
            print(f"ERROR: Could not ensure seed data via MCP: {e}")
            print("INFO: Continuing without seed data...")

        await self.update_current_graph_state_from_mcp()

    def add_llm_participant(self, name, model_name=None, client_type="anthropic"):
        for p in self.participants:
            if p["name"] == name:
                print(f"INFO: Participant '{name}' already exists.")
                return len(self.participants)

        default_model = os.environ.get("CLAUDE_MODEL")
        if client_type == "ollama":
            default_model = OLLAMA_MODEL

        participant = {
            "id": len(self.participants) + 1,
            "name": name,
            "type": "LLM",
            "model": model_name or default_model,
            "client_type": client_type,
            "contributions": []
        }
        self.participants.append(participant)
        print(f"INFO: Added participant: {name}")
        return len(self.participants)

    def add_agent_participant(self, name="ResearchAgent"):
        for p in self.participants:
            if p["name"] == name and p["type"] == "Agent":
                print(f"INFO: Agent participant '{name}' already exists.")
                return len(self.participants)

        agent = {
            "id": len(self.participants) + 1,
            "name": name,
            "type": "Agent",
            "contributions": []
        }
        self.participants.append(agent)
        print(f"INFO: Added Agent participant: {name}")
        return len(self.participants)

    async def get_graph_json_for_prompt(self):
        """Get the current graph state as JSON from the MCP server."""
        try:
            graph_data = await self.mcp_client.get_graph()
            return {
                "nodes": [{"id": node['id'], "labels": node['labels'], "properties": node['properties']} for node in graph_data.get("nodes", [])],
                "relationships": [{"id": rel['id'], "type": rel['type'], "source": rel['startNodeId'], "target": rel['endNodeId'], "properties": rel.get("properties", {})} for rel in graph_data.get("relationships", [])],
            }
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Could not fetch graph JSON from MCP: {e}")
            return {"error": str(e)}
        except httpx.RequestError as e:
            print(f"ERROR: Could not connect to MCP to fetch graph JSON: {e}")
            return {"error": str(e)}

    async def generate_prompt_for_llm(self, participant_index):
        """Generate a prompt for the LLM to contribute in JSON format, including the current graph."""
        participant = self.participants[participant_index]
        current_graph_json = await self.get_graph_json_for_prompt()
        existing_node_names = []
        if "nodes" in current_graph_json:
            existing_node_names = [node["properties"].get("name") for node in current_graph_json["nodes"] if "properties" in node and "name" in node["properties"]]

        existing_nodes_str = "\n- ".join(sorted([name for name in existing_node_names if name])) if existing_node_names else "No nodes yet."

        prompt = f"""
        # LLM Knowledge Graph Game - Turn {self.turn_count + 1}

        You are participating in a collaborative knowledge graph building game as {participant['name']}.

        ## Current Knowledge Graph Nodes

        The knowledge graph currently contains nodes with these names:
        - {existing_nodes_str}

        ## Game Rules

        Your primary task is to expand the existing knowledge graph, focusing not just on what is related, but also on **why** things are connected. When you propose new nodes or relationships, please tag each relationship in the 'properties' as either "semantic" or "causal" and assign it a 'weight' between 0.0 and 1.0 indicating the strength or likelihood of the connection. Also include an 'analysis' property for both new nodes and relationships.

        You can contribute by:

        1. Adding 1-3 new nodes connected to existing nodes.
        2. Adding new relationships between existing nodes.

        Your contribution should:
        - Be connected to at least one existing node in the graph (by name).
        - Provide novel, creative, and accurate information.
        - Explore interesting, non-obvious, yet relevant connections.
        - Reflect your unique perspective or "expertise" as {participant['name']}.

        ## Response Format

        Please respond with a JSON object:

        
        {{
          "explanation": "A brief explanation...",
          "nodes": [
            {{
              "name": "Name of new node 1",
              "labels": ["Category1", "Category2"],
              "properties": {{
                "description": "...",
                "analysis": "Analysis of this entity.",
                // ... other properties
              }}
            }},
            // ... more new nodes
          ],
          "relationships": [
            {{
              "source_node_name": "Name of existing node",
              "target_node_name": "Name of node",
              "type": "RELATIONSHIP_TYPE",
              "properties": {{
                "tag": "semantic" or "causal",
                "weight": 0.0 to 1.0,
                "analysis": "Analysis of the relationship.",
                // ... other properties
              }}
            }},
            // ... more new relationships
          ],
          "future_directions": ["Suggestion 1", "Suggestion 2"]
        }}
        

        Ensure that you reference existing nodes by their `"name"` property when defining relationships. Be insightful and aim to enrich the graph with valuable, interconnected knowledge!
        """
        return prompt

    def extract_json_from_response(self, response_text: str) -> Optional[dict]:
        """Extract JSON from response text, handling markdown code blocks."""
        import re
        
        # First try direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'(?:json)?\s*(\{.*?\})\s*'
        matches = re.findall(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON-like content without code blocks
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None

    async def process_llm_turn(self, participant_index):
        """Process a turn for an LLM participant, sending JSON to MCP server."""
        participant = self.participants[participant_index]
        prompt = await self.generate_prompt_for_llm(participant_index)
        llm_response_json = None

        try:
            if participant["client_type"] == "anthropic":
                if not self.client:
                    raise Exception("Anthropic client not available.")
                response = self.client.messages.create(
                    model=participant["model"],
                    max_tokens=4000,
                    temperature=0.8,
                    messages=[{"role": "user", "content": str(prompt)}]
                )
                response_text = response.content[0].text
                llm_response_json = self.extract_json_from_response(response_text)
                if not llm_response_json:
                    print(f"WARNING: Could not decode JSON from Anthropic for {participant['name']}: {response_text[:200]}...")
                    return {"participant": participant["name"], "success": False, "message": "Could not decode JSON response."}

            elif participant["client_type"] == "ollama":
                if not self.ollama_available or not self.ollama_client:
                    raise Exception("Ollama client not available.")
                print(f"INFO: Sending prompt to Ollama model: {participant['model']} for participant {participant['name']}")
                ollama_response = self.ollama_client.chat(
                    model=participant['model'],
                    messages=[{'role': 'user', 'content': str(prompt)}],
                    stream=False
                )
                if ollama_response and 'message' in ollama_response and 'content' in ollama_response['message']:
                    response_text = ollama_response['message']['content']
                    llm_response_json = self.extract_json_from_response(response_text)
                    if not llm_response_json:
                        print(f"WARNING: Could not decode JSON from Ollama for {participant['name']}: {response_text[:200]}...")
                        return {"participant": participant["name"], "success": False, "message": "Could not decode JSON response."}
                else:
                    raise Exception("Invalid or empty response structure from Ollama.")
            else:
                raise ValueError(f"Unknown client type: {participant['client_type']}")

            if llm_response_json:
                print(f"\nDEBUG: LLM ({participant['name']}) JSON response:\n'''\n{json.dumps(llm_response_json, indent=2)}\n'''")
                try:
                    # Send JSON to MCP server to update the graph
                    response = await self.mcp_client.execute_json_to_cypher(llm_response_json)
                    mcp_response = response.json()
                    if mcp_response.get("success"):
                        print(f"INFO: Graph updated via MCP for {participant['name']}.")
                        await self.update_current_graph_state_from_mcp()
                        self.turn_count += 1
                        contribution = {
                            "turn": self.turn_count,
                            "explanation": llm_response_json.get("explanation"),
                            "json_payload": llm_response_json,
                            "mcp_response": mcp_response,
                            "success": True,
                            "message": "Graph updated via MCP."
                        }
                        participant["contributions"].append(contribution)
                        return {"participant": participant["name"], "success": True, "message": "Graph updated via MCP.", "mcp_response": mcp_response, "llm_response": llm_response_json}
                    else:
                        print(f"WARNING: MCP server reported failure: {mcp_response.get('message')}")
                        return {"participant": participant["name"], "success": False, "message": f"MCP server failed to update graph: {mcp_response.get('message')}", "mcp_response": mcp_response, "llm_response": llm_response_json}
                except httpx.RequestError as e:
                    print(f"ERROR: Could not communicate with MCP server: {e}")
                    return {"participant": participant["name"], "success": False, "message": f"Error communicating with MCP server: {e}"}
                except httpx.HTTPStatusError as e:
                    print(f"HTTP error from MCP server: {e.response.status_code} - {e.response.text}")
                    return {"participant": participant["name"], "success": False, "message": f"HTTP error from MCP server: {e.response.status_code}."}
            else:
                return {"participant": participant["name"], "success": False, "message": "No valid JSON response from LLM."}

        except Exception as e:
            print(f"Error processing LLM turn for {participant['name']}: {e}")
            return {"participant": participant["name"], "success": False, "message": f"Error processing LLM turn: {e}"}

    async def run_game(self, num_turns=5):
        """Run the game for a specified number of turns"""
        if not self.participants:
            print("No participants added. Add participants before running the game.")
            return []

        print("INFO: Starting game...")
        results = []
        for turn in range(num_turns):
            print(f"\n--- Turn {turn + 1} ---")
            participant_index = self.turn_count % len(self.participants)
            participant = self.participants[participant_index]
            print(f"Participant: {participant['name']}")
            result = await self.process_llm_turn(participant_index)
            if result and result.get("success"):
                print(f"Contribution successful: {result['message']}")
            elif result:
                print(f"Contribution failed: {result['message']}")
            results.append(result)
            time.sleep(1) # Be gentle on the LLMs and server

        print("\n--- Game Complete ---")
        return results

    async def get_graph_for_export(self):
        """Get the current graph state as JSON from the MCP server for export."""
        try:
            return await self.mcp_client.get_graph()
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Could not fetch graph JSON from MCP for export: {e}")
            return {"error": str(e)}
        except httpx.RequestError as e:
            print(f"ERROR: Could not connect to MCP to fetch graph JSON for export: {e}")
            return {"error": str(e)}

    async def export_graph(self, format="json"):
        """Export the current graph state."""
        graph_data = await self.mcp_client.get_graph()
        nodes_list = []
        for node in graph_data.get("nodes", []):
            nodes_list.append({"id": node['id'], "labels": node['labels'], "properties": node['properties']})

        relationships_list = []
        for rel in graph_data.get("relationships", []):
            relationships_list.append({
                "source": rel['startNodeId'],
                "target": rel['endNodeId'],
                "type": rel['type'],
                "properties": rel['properties']
            })

        if format == "json":
            return {"nodes": nodes_list, "relationships": relationships_list}
        elif format == "cypher":
            try:
                response = await self.mcp_client.execute_query("CALL apoc.export.cypher.all(null, {config: {}}) YIELD output")
                if response and len(response) > 0 and 'output' in response[0]:
                    # The output might be a single string or a list
                    if isinstance(response[0]['output'], list):
                        return "\n".join(response[0]['output'])
                    else:
                        return response[0]['output']
                else:
                    return "Could not export to Cypher using APOC."
            except httpx.HTTPStatusError as e:
                print(f"Error exporting to Cypher via APOC: {e.response.status_code} - {e.response.text}")
                return f"Error exporting to Cypher: {e.response.status_code}"
            except httpx.RequestError as e:
                print(f"Error connecting to MCP for Cypher export: {e}")
                return "Error connecting to MCP for Cypher export."
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def close(self):
        """Close Neo4j driver connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
        print("INFO: Neo4j driver connection closed.")

# Example usage and main game execution block
if __name__ == "__main__":
    # Ensure environment variables are set
    if not all([os.environ.get(key) for key in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "ANTHROPIC_API_KEY", "OLLAMA_MODEL", "MCP_SERVER_URL"]]):
        print("Error: Please set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ANTHROPIC_API_KEY (optional), OLLAMA_MODEL, and MCP_SERVER_URL environment variables.")
    else:
        game = KnowledgeGraphGame()

        # Add LLM participants
        game.add_llm_participant("PhilosopherOfComplexity_LLM", model_name="llama3:latest", client_type="ollama")
        game.add_llm_participant("RoboticsFuturist_LLM", model_name="llama3:latest", client_type="ollama")
        # Add more participants as needed

        num_participants = len(game.participants)
        if num_participants > 0:
            print(f"INFO: Starting game with {num_participants} participants for {num_participants * 2} turns.")
            results = asyncio.run(game.run_game(num_turns=num_participants * 2))
            print("\n--- Game Results ---")
            for result in results:
                print(f"{result.get('participant')}: {result.get('message')}")
        else:
            print("WARNING: No participants added to the game. Game will not run.")

        final_graph_json = asyncio.run(game.export_graph(format="json"))
        print("\n--- Final Graph (JSON Representation) ---")
        print(final_graph_json)

        game.close()
        print("INFO: Game script finished.")