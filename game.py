import os
from neo4j import GraphDatabase
from anthropic import Anthropic
import re # Added missing import
import json
import random
import dotenv
import time # Add missing time import
import redis
import certifi # Import certifi
import hashlib # Moved import to top

dotenv.load_dotenv()

# --- Attempt to set SSL_CERT_FILE programmatically using certifi ---
# This is crucial for ensuring Python can find SSL certificates, especially on macOS
try:
    certifi_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi_path
    os.environ['REQUESTS_CA_BUNDLE'] = certifi_path # For other libs like requests
    print(f"INFO: Programmatically set SSL_CERT_FILE to: {certifi_path}")
except ImportError:
    print("WARNING: certifi package not found. SSL connections might fail. Please install certifi.")
except Exception as e:
    print(f"ERROR: Could not set SSL_CERT_FILE from certifi: {e}")
# --- End of SSL_CERT_FILE setting ---



# Neo4j setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Redis setup
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Anthropic API setup
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # Update as needed

class KnowledgeGraphGame:
    def __init__(self):
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD if REDIS_PASSWORD else None,
                db=REDIS_DB,
                decode_responses=True  # Automatically decode responses to strings
            )
            self.redis.ping()  # Test connection
            self.redis_available = True
            print("Connected to Redis successfully.")
        except redis.ConnectionError:
            self.redis_available = False
            print("WARNING: Could not connect to Redis. Caching will be disabled.")
        
        # Initialize Anthropic client if key is available
        if ANTHROPIC_API_KEY:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self.client = None
            print("WARNING: ANTHROPIC_API_KEY not set. LLM functionality will not work.")
        
        # Initialize game state
        self.turn_count = 0
        self.participants = []
        self.current_graph_state = {}
        
        # Try to load game state from Redis if available
        self.load_game_state_from_redis()
        # If after loading from Redis, current_graph_state is still empty,
        # (e.g. Redis was down, or it's the very first run with an empty Redis)
        # then populate it from the Neo4j database.
        if not self.current_graph_state:
            print("INFO: current_graph_state is empty after attempting Redis load. Updating from Neo4j DB.")
            self.update_graph_state() # This will also save to Redis if available
    
    def load_game_state_from_redis(self):
        """Load game state from Redis if available"""
        if not self.redis_available:
            return False
        
        try:
            # Load turn count
            turn_count = self.redis.get("game:turn_count")
            if turn_count:
                self.turn_count = int(turn_count)
            
            # Load participants
            participants_json = self.redis.get("game:participants")
            if participants_json:
                self.participants = json.loads(participants_json)
            
            # Load graph state
            graph_state_json = self.redis.get("game:graph_state")
            if graph_state_json:
                self.current_graph_state = json.loads(graph_state_json)
            
            if turn_count or participants_json or graph_state_json:
                print("INFO: Game state loaded from Redis.")
            return True
        except Exception as e:
            print(f"Error loading game state from Redis: {str(e)}")
            return False
    
    def save_game_state_to_redis(self):
        """Save current game state to Redis"""
        if not self.redis_available:
            return False
        
        try:
            # Save turn count
            self.redis.set("game:turn_count", self.turn_count)
            
            # Save participants
            self.redis.set("game:participants", json.dumps(self.participants))
            
            # Save graph state
            self.redis.set("game:graph_state", json.dumps(self.current_graph_state))
            
            print("INFO: Game state saved to Redis.")
            return True
        except Exception as e:
            print(f"Error saving game state to Redis: {str(e)}")
            return False
    
    def cache_llm_response(self, prompt, response, participant_name, model):
        """Cache an LLM response in Redis"""
        if not self.redis_available:
            return False
        
        try:
            # Create a cache key based on prompt hash
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cache_key = f"llm:response:{prompt_hash}"
            
            # Store the response with metadata
            cache_data = {
                "response": response,
                "participant": participant_name,
                "model": model,
                "timestamp": json.dumps({"$date": int(time.time() * 1000)})
            }
            
            # Store in Redis with expiration (e.g., 24 hours)
            self.redis.setex(
                cache_key,
                86400,  # 24 hours in seconds
                json.dumps(cache_data)
            )
            
            return True
        except Exception as e:
            print(f"Error caching LLM response: {str(e)}")
            return False
    
    def get_cached_llm_response(self, prompt):
        """Get a cached LLM response from Redis if available"""
        if not self.redis_available:
            return None
        
        try:
            # Create the same cache key based on prompt hash
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cache_key = f"llm:response:{prompt_hash}"
            
            # Try to get from Redis
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
        except Exception as e:
            print(f"Error retrieving cached LLM response: {str(e)}")
            return None
    
    def initialize_graph(self, force_reset=False):
        """
        Ensure the foundational knowledge graph nodes and relationships exist.
        If force_reset is True, clears the entire graph first.
        Uses MERGE to be idempotent for the seed data.
        """
        with self.driver.session() as session:
            if force_reset:
                print("INFO: Force reset requested. Clearing entire graph...")
                session.run("MATCH (n) DETACH DELETE n")
            
            print("INFO: Ensuring foundational seed nodes and relationships exist using MERGE...")
            # Use MERGE for initial nodes and relationships to make it idempotent
            session.run("""
            MERGE (biomimeticAlgo:Concept:ComputerScience {name: "Biomimetic Algorithms"})
            ON CREATE SET
              biomimeticAlgo.description = "Algorithms inspired by biological processes in nature",
              biomimeticAlgo.domain = "Computer Science"
            MERGE (antColony:Concept:Biology {name: "Ant Colony Optimization"})
            ON CREATE SET
              antColony.description = "Swarm intelligence based on pheromone trail optimization",
              antColony.domain = "Biology"
            MERGE (networkRouting:Concept:Telecommunications {name: "Network Routing Algorithms"})
            ON CREATE SET
              networkRouting.description = "Methods for determining optimal paths in communication networks",
              networkRouting.domain = "Telecommunications"
            MERGE (collectiveIntelligence:Concept:Behavior {name: "Collective Intelligence"})
            ON CREATE SET
              collectiveIntelligence.description = "Shared or group intelligence emerging from collaboration",
              collectiveIntelligence.domain = "Complex Systems"

            MERGE (biomimeticAlgo)-[r1:INSPIRED_BY]->(collectiveIntelligence)
            ON CREATE SET r1.description = "Takes design principles from"
            MERGE (antColony)-[r2:DEMONSTRATES]->(collectiveIntelligence)
            ON CREATE SET r2.description = "Shows patterns of"
            MERGE (biomimeticAlgo)-[r3:APPLIED_TO]->(networkRouting)
            ON CREATE SET r3.description = "Used to optimize"
            MERGE (antColony)-[r4:SERVES_AS_MODEL_FOR]->(biomimeticAlgo)
            ON CREATE SET r4.description = "Provides algorithmic framework for"
            """)
            
        self.update_graph_state() # Refresh in-memory state and save to Redis
        if force_reset:
            print("Knowledge graph reset and initialized with starting seed.")
        else:
            print("Foundational seed nodes and relationships ensured (created if not existing).")
        
    def update_graph_state(self):
        """Update the current state of the graph by querying Neo4j"""
        with self.driver.session() as session:
            # Get all nodes
            nodes_result = session.run("""
            MATCH (n)
            RETURN n, labels(n) as labels, elementId(n) as element_id
            """)
            
            nodes = {}
            for record in nodes_result:
                node = record["n"]
                node_id = record["element_id"] # Use element_id
                node_labels = record["labels"]
                
                nodes[node_id] = {
                    "id": node_id, # This 'id' is now the elementId string
                    "labels": node_labels,
                    **dict(node)
                }
            
            # Get all relationships
            rels_result = session.run("""
            MATCH ()-[r]->()
            RETURN elementId(r) as element_id, type(r) as type, elementId(startNode(r)) as start_node_element_id, elementId(endNode(r)) as end_node_element_id, properties(r) as props
            """)
            
            relationships = {}
            for record in rels_result:
                rel_id = record["element_id"] # Use element_id
                relationships[rel_id] = {
                    "id": rel_id, # This 'id' is now the elementId string
                    "type": record["type"],
                    "start": record["start_node_element_id"],
                    "end": record["end_node_element_id"],
                    "properties": record["props"]
                }
            
            self.current_graph_state = {
                "nodes": nodes,
                "relationships": relationships
            }
            
            # Save updated state to Redis
            self.save_game_state_to_redis()
    
    def add_llm_participant(self, name, model_name=None):
        """Add an LLM participant to the game"""
        # Check if participant with this name already exists to avoid duplicates if loading from Redis
        for p in self.participants:
            if p["name"] == name:
                print(f"INFO: Participant '{name}' already exists. Not adding again.")
                return len(self.participants)

        participant = {
            "id": len(self.participants) + 1, # This ID might not be unique if participants are loaded then more added
            "name": name,
            "type": "LLM",
            "model": model_name or CLAUDE_MODEL,
            "contributions": []
        }
        self.participants.append(participant)
        print(f"INFO: Added participant: {name}")
        
        # Save updated participants to Redis
        self.save_game_state_to_redis()
        
        return len(self.participants)
    
    def get_graph_cypher_representation(self):
        """Get the current graph state as a series of Cypher commands"""
        with self.driver.session() as session:
            # Get all nodes
            nodes_result = session.run("""
            MATCH (n)
            RETURN n, labels(n) as labels, elementId(n) as element_id
            """)
            
            cypher_commands = ["// Nodes"]
            for record in nodes_result:
                node = record["n"]
                node_labels = record["labels"]
                
                # Format properties
                props = dict(node) # Make a mutable copy
                props_str_parts = []
                for k, v_val in props.items():
                    if isinstance(v_val, str):
                        escaped_v = v_val.replace("'", "\\'") # Escape single quotes
                        props_str_parts.append(f"{k}: '{escaped_v}'")
                    elif isinstance(v_val, bool):
                        props_str_parts.append(f"{k}: {str(v_val).lower()}") # Cypher uses lowercase true/false
                    else: # Numbers, etc.
                        props_str_parts.append(f"{k}: {v_val}")
                props_str = ", ".join(props_str_parts)
                
                # Create node creation command
                labels_str = ":" + ":".join(node_labels) if node_labels else ""
                cypher_commands.append(f"// Node with elementId: {record['element_id']}\nCREATE (node{labels_str} {{{props_str}}})")
            
            # Get all relationships
            cypher_commands.append("\n// Relationships")
            rels_result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN elementId(a) as a_element_id, type(r) as type, properties(r) as props, elementId(b) as b_element_id
            """)
            
            for record in rels_result:
                a_element_id = record["a_element_id"] 
                b_element_id = record["b_element_id"] 
                rel_type = record["type"]
                props = record["props"] 

                if props:
                    props_str_parts = []
                    for k, v_val in props.items():
                        if isinstance(v_val, str):
                            escaped_v = v_val.replace("'", "\\'")
                            props_str_parts.append(f"{k}: '{escaped_v}'")
                        elif isinstance(v_val, bool):
                            props_str_parts.append(f"{k}: {str(v_val).lower()}")
                        else: # Numbers, etc.
                            props_str_parts.append(f"{k}: {v_val}")
                    props_str = " {" + ", ".join(props_str_parts) + "}"
                else:
                    props_str = ""
                
                cypher_commands.append(f"// Relationship from elementId {a_element_id} to {b_element_id}\n// MATCH (nodeA), (nodeB) WHERE elementId(nodeA) = '{a_element_id}' AND elementId(nodeB) = '{b_element_id}' CREATE (nodeA)-[:{rel_type}{props_str}]->(nodeB)")
            
            return "\n".join(cypher_commands)
    
    def generate_prompt_for_llm(self, participant_index):
        """Generate a prompt for the LLM to contribute to the knowledge graph"""
        participant = self.participants[participant_index]
        
        # Get the current graph state
        cypher_representation = self.get_graph_cypher_representation()
        
        # Generate a prompt based on game rules
        prompt = f"""
        # LLM Knowledge Graph Game - Turn {self.turn_count + 1}
        
        You are participating in a collaborative knowledge graph building game as {participant['name']}.
        
        ## Current Knowledge Graph
        
        The knowledge graph currently contains these nodes and relationships (in Cypher format):
        
        
        {cypher_representation}
        
        
        ## Game Rules
        
        Your primary task is to expand the existing knowledge graph.
        - **Identify existing nodes** from the "Current Knowledge Graph" that you want to connect your new information to. You can identify them by their properties (e.g., name).
        - Then, make ONE of the following types of contributions:
        
        1. 1-3 new nodes connected to existing nodes
        2. New relationships between existing nodes
        3. Labels to nodes when recognizing patterns
        4. Properties to existing nodes or relationships
        
        **Important for Cypher:**
        - To connect to an existing node, use a `MATCH` clause. For example: `MATCH (existingNode:Concept {{name: "Ant Colony Optimization"}})`
        - Then, use `CREATE` or `MERGE` for your new additions, linking them to `existingNode`.
        - Your Cypher commands should ONLY include the new elements or modifications you are adding. Do NOT repeat the Cypher for the existing graph.
        
        Your contribution should:
        - Be connected to at least one existing element in the graph
        - Provide novel, creative, and accurate information
        - Explore interesting, non-obvious, yet relevant connections
        - Reflect your unique perspective or "expertise" as {participant['name']}
        
        ## Response Format
        
        Please respond with:
        
        1. A brief explanation of what you're adding and why it's interesting (2-3 sentences)
        2. The Cypher commands for **your additions only**, using the following format:
        
        ```cypher
        YOUR CYPHER COMMANDS FOR ADDITIONS HERE
        ```
        
        3. Suggest 1-2 potential directions for future participants to explore
        
        Be insightful and aim to enrich the graph with valuable, interconnected knowledge!
        """
        
        return prompt
    
    def process_llm_turn(self, participant_index):
        """Process a turn for an LLM participant"""
        if not self.client:
            print("No Anthropic client available. Cannot process LLM turn.")
            return None
        
        participant = self.participants[participant_index]
        prompt = self.generate_prompt_for_llm(participant_index)
        
        try:
            # --- Call Claude API with Retry Logic ---
            max_retries = 3
            retry_delay = 5  # seconds
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model=participant["model"],
                        max_tokens=4000,
                        temperature=0.8,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    break # Success, exit retry loop
                except (self.client.APIConnectionError, self.client.RateLimitError, self.client.APIStatusError) as e:
                    # APIStatusError will catch 529s
                    print(f"WARNING: API error for {participant['name']} (Attempt {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2 # Exponential backoff
                    else:
                        print(f"ERROR: Max retries reached for {participant['name']}. API call failed.")
                        raise # Re-raise the last exception to be caught by the outer try-except
            
            if response is None: # Should be caught by re-raise above, but as a safeguard
                raise Exception("API response was None after retries, indicating failure.")
            # --- End of API Call with Retry Logic ---

            
            print(f"\nDEBUG: LLM ({participant['name']}) raw response text:\n'''\n{response.content[0].text}\n'''")
            
            response_text = response.content[0].text
            
            # Extract Cypher commands from response
            # Regex to capture content within a markdown Cypher block ```cypher ... ```
            # or after a specific comment line // Your Cypher commands here
            cypher_block_match = re.search(
                r"```cypher\n(.*?)\n```|// Your Cypher commands here\n(.*?)(?=\n\n\n3\.|\Z)",
                response_text,
                re.DOTALL | re.IGNORECASE)
            
            cypher_commands = None
            if cypher_block_match:
                if cypher_block_match.group(1): # Markdown block matched
                    cypher_commands = cypher_block_match.group(1).strip()
                elif cypher_block_match.group(2): # Comment block matched
                    cypher_commands = cypher_block_match.group(2).strip()
            
            if cypher_commands: # If cypher_commands is not None and not empty after strip
                print(f"DEBUG: Extracted Cypher for {participant['name']}:\n'''\n{cypher_commands}\n'''")
                
                # Execute Cypher commands
                success, message = self.execute_cypher_commands(cypher_commands)
                
                # Record the contribution
                contribution = {
                    "turn": self.turn_count,
                    "explanation": response_text, # Store the full response for context
                    "cypher": cypher_commands,
                    "success": success,
                    "message": message
                }
                participant["contributions"].append(contribution)
                print(f"DEBUG: Cypher execution for {participant['name']} - Success: {success}, Message: {message}")
                
                # Update game state
                if success:
                    self.update_graph_state() # This will also save to Redis
                    self.turn_count += 1
                    self.save_game_state_to_redis() # Explicitly save turn_count if it changed
                
                return {
                    "participant": participant["name"],
                    "success": success,
                    "message": message,
                    "response": response_text,
                    "cypher": cypher_commands
                }
            else: # No Cypher block found or it was empty
                print(f"DEBUG: Could not extract valid Cypher commands for {participant['name']}.")
                return {
                    "participant": participant["name"],
                    "success": False,
                    "message": "Could not extract Cypher commands from LLM response or the block was empty.",
                    "response": response_text,
                    "cypher": None
                }
        except Exception as e:
            print(f"Error processing LLM turn for {participant['name']}: {str(e)}")
            return {
                "participant": participant["name"],
                "success": False,
                "message": f"Error: {str(e)}",
                "response": None, 
                "cypher": None
            }
    
    def execute_cypher_commands(self, cypher_commands):
        """Execute a series of Cypher commands safely"""
        try:
            with self.driver.session() as session:
                commands = [cmd.strip() for cmd in cypher_commands.split(';') if cmd.strip()]
                
                if not commands:
                    return False, "No valid Cypher commands to execute after splitting and stripping."

                for cmd in commands:
                    print(f"DEBUG: Executing Cypher command: {cmd}")
                    session.run(cmd)
                
            return True, "Cypher commands executed successfully"
        except Exception as e:
            print(f"DEBUG: Error during Cypher execution: {str(e)}")
            return False, f"Error executing Cypher commands: {str(e)}"
    
    def run_game(self, num_turns=5):
        """Run the game for a specified number of turns"""
        if not self.participants:
            print("No participants added. Add participants before running the game.")
            return [] # Return empty list if no participants
        
        # Initialize the graph if not already done (e.g., if Redis state wasn't loaded and current_graph_state is empty)
        if not self.current_graph_state: # Check if graph state was loaded from Redis or initialized
            print("INFO: No existing graph state found. Initializing graph...")
            self.initialize_graph()
        
        results = []
        
        for turn_attempt in range(num_turns): 
            print(f"\n--- Attempting Turn {turn_attempt + 1} (Overall Successful Turns So Far: {self.turn_count}) ---")
            
            if not self.participants: 
                print("Error: No participants available to take a turn.")
                break

            participant_index = self.turn_count % len(self.participants)
            participant = self.participants[participant_index]
            print(f"Participant: {participant['name']}")
            
            result = None 
            if participant["type"] == "LLM":
                result = self.process_llm_turn(participant_index)
                if result and result.get("success"):
                    print(f"Contribution successful: {result['message']}")
                elif result:
                    print(f"Contribution failed: {result['message']}")
                else: 
                    print(f"Contribution failed: No result from process_llm_turn for {participant['name']}")
                
                if result: 
                    results.append(result)
            
            if not result or not result.get("success"):
                print(f"INFO: Turn for {participant['name']} was not successful or did not yield changes. Graph state remains as per last successful update.")
        
        print("\n--- Game Complete ---")
        self.save_game_state_to_redis() 
        return results
    
    def export_graph(self, format="json"):
        """Export the current graph"""
        if format == "json":
            return json.dumps(self.current_graph_state, indent=2)
        elif format == "cypher":
            return self.get_graph_cypher_representation()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'driver'):
            self.driver.close()
        print("INFO: Neo4j driver connection closed.")

# Example usage and main game execution block
if __name__ == "__main__":
    game = KnowledgeGraphGame()
    
    # Add LLM participants
    # Using descriptive names and different Claude models if available/desired
    game.add_llm_participant("PhilosopherOfComplexity_LLM", model_name="claude-3-5-sonnet-20240620") 
    game.add_llm_participant("RoboticsFuturist_LLM", model_name="claude-3-5-sonnet-20240620")
    game.add_llm_participant("ComputationalBiologist_LLM", model_name="claude-3-5-sonnet-20240620")
    game.add_llm_participant("Neuroscientist_LLM", model_name="claude-3-5-sonnet-20240620") 
    game.add_llm_participant("MaterialsInnovator_LLM", model_name="claude-3-5-sonnet-20240620")
    game.add_llm_participant("AI_Ethicist_LLM", model_name="claude-3-5-sonnet-20240620") # Opus for deeper ethical reasoning if available
    game.add_llm_participant("EvolutionaryEcologist_LLM", model_name="claude-3-5-sonnet-20240620")
    
    # Initialize graph: ensures seed nodes exist using MERGE, doesn't wipe by default.
    # Set force_reset=True if you want to start completely fresh (e.g., game.initialize_graph(force_reset=True))
    # The __init__ method already attempts to load state from Redis and then syncs from DB if Redis state is empty.
    # This call ensures the seed nodes are present on top of whatever was loaded or exists.
    game.initialize_graph(force_reset=False) 
    
    num_participants = len(game.participants)
    if num_participants > 0:
        print(f"INFO: Starting game with {num_participants} participants for {num_participants * 2} turns.")
        results = game.run_game(num_turns=num_participants * 2) 
        
    else:
        print("WARNING: No participants added to the game. Game will not run.")
    
    if game.current_graph_state: 
        final_graph_cypher = game.export_graph(format="cypher")
        print("\n--- Final Graph (Cypher Representation) ---")
        print(final_graph_cypher)
    else:
        print("INFO: No graph state to export.")
    
    game.close()
    print("INFO: Game script finished.")