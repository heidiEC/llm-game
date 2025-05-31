import os
import json
import time
import re
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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")

class KnowledgeGraphGame:
    def __init__(self):
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

    def setup_sustainable_urban_development_participants(self):
        """Set up participants for Sustainable Urban Development domain"""
        participants_config = [
            {
                "name": "UrbanPlanningExpert_LLM",
                "expertise": "Urban Planning & Design",
                "focus": "Zoning, transportation networks, mixed-use development, smart city infrastructure"
            },
            {
                "name": "EnvironmentalScientist_LLM", 
                "expertise": "Environmental Systems",
                "focus": "Air quality, water management, green infrastructure, carbon footprint, biodiversity"
            },
            {
                "name": "SocialPolicyAnalyst_LLM",
                "expertise": "Social Policy & Equity",
                "focus": "Housing affordability, community engagement, social justice, demographic impacts"
            },
            {
                "name": "EconomicDevelopmentSpecialist_LLM",
                "expertise": "Economic Systems",
                "focus": "Local economy, job creation, cost-benefit analysis, funding mechanisms"
            },
            {
                "name": "TechnologyInnovator_LLM",
                "expertise": "Smart City Technology",
                "focus": "IoT sensors, data analytics, renewable energy systems, digital governance"
            },
            {
                "name": "TransportationEngineer_LLM",
                "expertise": "Mobility & Transportation",
                "focus": "Public transit, cycling infrastructure, electric vehicles, traffic optimization"
            },
            {
                "name": "AffordableHousingAdvocate_LLM",
                "expertise": "Affordable Housing",
                "focus": "Housing policy, gentrification, tenant rights, community land trusts"
            },
            {
                "name": "CommunityAdvocate_LLM",
                "expertise": "Community Engagement",
                "focus": "Resident participation, cultural preservation, neighborhood dynamics, grassroots initiatives"
            },
            {
                "name": "ClimateResilienceExpert_LLM",
                "expertise": "Climate Adaptation",
                "focus": "Flood management, heat island effects, disaster preparedness, resilient infrastructure"
            },
            {
                "name": "PublicHealthSpecialist_LLM",
                "expertise": "Public Health",
                "focus": "Air quality, walkability, mental health, health equity, environmental health, climate change, infectious disease prevention, health policy, community health"
            },
            {
                "name": "DisasterRecoveryExpert_LLM",
                "expertise": "Disaster Resilience & Emergency Management",
                "focus": "Emergency preparedness, evacuation planning, infrastructure hardening, community resilience, post-disaster recovery"
            }
        ]
        
        for participant in participants_config:
            self.add_llm_participant(
                participant["name"], 
                model_name="llama3:latest", 
                client_type="ollama"
            )
            # Store expertise info for enhanced prompts
            self.participants[-1]["expertise"] = participant["expertise"]
            self.participants[-1]["focus"] = participant["focus"]

    async def initialize_graph(self, force_reset=False, domain="sustainable_urban_development"):
        """Enhanced initialization with domain-specific seed data"""
        if force_reset:
            print("INFO: Force reset requested. Clearing entire graph via MCP...")
            try:
                response = await self.mcp_client.execute_query("MATCH (n) DETACH DELETE n")
                print("INFO: Graph cleared via MCP.")
            except Exception as e:
                print(f"ERROR: Could not clear graph via MCP: {e}")

        print(f"INFO: Ensuring foundational seed nodes and relationships exist for {domain}...")
        
        if domain == "sustainable_urban_development":
            seed_cypher = """
            MERGE (sustainableUrbanDev:Domain {name: "Sustainable Urban Development", description: "Integrated approach to creating environmentally responsible, economically viable, and socially equitable urban environments"});
            
            MERGE (urbanPlanning:TaxonomyCategory:UrbanPlanning {name: "Urban Planning", description: "Strategic design and organization of urban spaces", domain: "Sustainable Urban Development"});
            MERGE (environmentalSystems:TaxonomyCategory:Environmental {name: "Environmental Systems", description: "Natural and built environmental interactions in urban contexts", domain: "Sustainable Urban Development"});
            MERGE (socialEquity:TaxonomyCategory:Social {name: "Social Equity", description: "Fair distribution of resources and opportunities across urban populations", domain: "Sustainable Urban Development"});
            MERGE (economicDevelopment:TaxonomyCategory:Economic {name: "Economic Development", description: "Sustainable economic growth and prosperity in urban areas", domain: "Sustainable Urban Development"});
            MERGE (smartTechnology:TaxonomyCategory:Technology {name: "Smart Technology", description: "Digital and technological solutions for urban challenges", domain: "Sustainable Urban Development"});
            MERGE (transportation:TaxonomyCategory:Transportation {name: "Transportation Systems", description: "Mobility infrastructure and services in urban environments", domain: "Sustainable Urban Development"});
            MERGE (climateResilience:TaxonomyCategory:Climate {name: "Climate Resilience", description: "Urban adaptation and mitigation strategies for climate change", domain: "Sustainable Urban Development"});
            
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(urbanPlanning);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(environmentalSystems);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(socialEquity);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(economicDevelopment);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(smartTechnology);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(transportation);
            MERGE (sustainableUrbanDev)-[:ENCOMPASSES {weight: 1.0, tag: "semantic"}]->(climateResilience);
            
            MERGE (urbanPlanning)-[:INFLUENCES {weight: 0.9, tag: "causal", analysis: "Urban planning decisions directly shape environmental outcomes through land use and infrastructure choices"}]->(environmentalSystems);
            MERGE (transportation)-[:IMPACTS {weight: 0.8, tag: "causal", analysis: "Transportation systems significantly affect air quality, energy consumption, and urban form"}]->(environmentalSystems);
            MERGE (economicDevelopment)-[:AFFECTS {weight: 0.7, tag: "causal", analysis: "Economic policies and development patterns influence social equity through job access and housing affordability"}]->(socialEquity);
            MERGE (smartTechnology)-[:ENABLES {weight: 0.8, tag: "causal", analysis: "Smart city technologies can optimize resource use and improve service delivery across all urban systems"}]->(urbanPlanning);
            """
        else:
            # Default seed data
            seed_cypher = """
            MERGE (biomimeticAlgo:Concept:ComputerScience {name: "Biomimetic Algorithms", description: "Algorithms inspired by biological processes in nature", domain: "Computer Science"});
            MERGE (antColony:Concept:Biology {name: "Ant Colony Optimization", description: "Swarm intelligence based on pheromone trail optimization", domain: "Biology"});
            MERGE (networkRouting:Concept:Telecommunications {name: "Network Routing Algorithms", description: "Methods for determining optimal paths in communication networks", domain: "Telecommunications"});
            MERGE (collectiveIntelligence:Concept:Behavior {name: "Collective Intelligence", description: "Shared or group intelligence emerging from collaboration", domain: "Complex Systems"});
            MERGE (biomimeticAlgo)-[:INSPIRED_BY {analysis: "Biomimetic algorithms take inspiration from the principles of collective intelligence found in biological systems."}]->(collectiveIntelligence);
            MERGE (antColony)-[:DEMONSTRATES {analysis: "Ant colony optimization is a specific example of a system that demonstrates collective intelligence."}]->(collectiveIntelligence);
            MERGE (biomimeticAlgo)-[:APPLIED_TO {analysis: "Biomimetic algorithms, such as ant colony optimization, are applied to solve problems like network routing."}]->(networkRouting);
            MERGE (antColony)-[:SERVES_AS_MODEL_FOR {analysis: "The behavior of ant colonies serves as a model for the design of biomimetic algorithms like ACO."}]->(biomimeticAlgo);
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
        """Generate a domain and expertise-aware prompt for the LLM"""
        participant = self.participants[participant_index]
        current_graph_json = await self.get_graph_json_for_prompt()
        existing_node_names = []
        if "nodes" in current_graph_json:
            existing_node_names = [node["properties"].get("name") for node in current_graph_json["nodes"] if "properties" in node and "name" in node["properties"]]

        existing_nodes_str = "\n- ".join(sorted([name for name in existing_node_names if name])) if existing_node_names else "No nodes yet."

        expertise_section = ""
        if "expertise" in participant:
            expertise_section = f"""
## Your Expertise: {participant['expertise']}

Your area of specialization: {participant.get('focus', 'General expertise in this domain')}

While you should primarily contribute nodes and insights from your area of expertise, you are strongly encouraged to:
1. **Create cross-domain relationships** - Connect your concepts to nodes created by other experts
2. **Identify causal chains** - Look for cause-and-effect relationships, especially across disciplines
3. **Bridge knowledge gaps** - Help connect isolated concepts from different domains
4. **Include population segments and life contexts** - Consider specific groups of people and places where policies have real impact
"""

        prompt = f"""
# LLM Knowledge Graph Game - Turn {self.turn_count + 1}
## Domain: Sustainable Urban Development

You are participating in a collaborative knowledge graph building game as {participant['name']}.
{expertise_section}

## Current Knowledge Graph Nodes

The knowledge graph currently contains nodes with these names:
- {existing_nodes_str}

## CRITICAL: Response Format Requirements

**YOU MUST RESPOND WITH VALID JSON ONLY. NO MARKDOWN, NO EXPLANATIONS OUTSIDE THE JSON.**

Your response must be a single JSON object that starts with {{ and ends with }}. Do not include any text before or after the JSON.

Example format:
{{
  "explanation": "Brief explanation...",
  "nodes": [...],
  "relationships": [...],
  "future_directions": [...]
}}

## Game Rules & Objectives

Your primary task is to expand the knowledge graph with a focus on **cross-domain connections** and **causal relationships**. 

**Key Priorities:**
1. **Causal Relationships**: Identify cause-and-effect chains (tag as "causal" with weight 0.1-1.0)
2. **Semantic Relationships**: Create conceptual connections (tag as "semantic" with weight 0.1-1.0)
3. **Cross-Domain Bridges**: Connect your expertise area to other domains represented in the graph
4. **Population & Context Focus**: Include specific population groups and life settings (work, home, recreation)
5. **Practical Applications**: Focus on real-world, actionable concepts

**Important**: Consider including specific **population groups** and **life contexts** that are central to your expertise:
- Population segments: "Working families", "Elderly residents", "Low-income families", "College students"
- Life contexts: "Affordable housing complexes", "Public transit hubs", "Community centers", "Industrial zones"
- Specific policies/programs: "Section 8 housing", "Complete streets policies", "Green building standards"

**Contribution Guidelines:**
- Add 1-3 new nodes, preferably from your expertise area
- Create 2-4 relationships (mix of new connections and bridges to existing nodes)
- Weight relationships: 0.8-1.0 (strong), 0.5-0.7 (moderate), 0.1-0.4 (weak)
- Provide detailed analysis explaining WHY relationships exist

## JSON Response Format (COPY THIS STRUCTURE EXACTLY)

{{
  "explanation": "Brief explanation of your contribution and cross-domain connections...",
  "nodes": [
    {{
      "name": "Specific Node Name",
      "labels": ["PrimaryCategory", "SecondaryCategory"],
      "properties": {{
        "description": "Detailed description...",
        "domain": "Sustainable Urban Development",
        "expertise_area": "{participant.get('expertise', 'General')}",
        "analysis": "Why this concept is important and how it connects to the broader domain...",
        "practical_applications": "Real-world applications or examples..."
      }}
    }}
  ],
  "relationships": [
    {{
      "source_node_name": "Existing Node Name",
      "target_node_name": "New or Existing Node Name", 
      "type": "RELATIONSHIP_TYPE",
      "properties": {{
        "tag": "causal",
        "weight": 0.8,
        "analysis": "Detailed explanation of WHY this relationship exists and its strength...",
        "cross_domain": true,
        "evidence": "Supporting evidence or examples..."
      }}
    }}
  ],
  "future_directions": ["Specific suggestions for other experts to explore..."]
}}

REMEMBER: Respond with ONLY the JSON object. No markdown formatting, no explanations outside the JSON.
"""
        return prompt
    
    def extract_json_from_response(self, response_text: str) -> Optional[dict]:
        """Extract JSON from response text, handling markdown code blocks and various formats."""
        import re
        
        # First try direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'(?:json)?\s*(\{.*?\})\s*',  #  { } 
            r'(\{.*?\})',                 #  { } 
            r'`(\{.*?\})`',                     # ` { } `
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    parsed_json = json.loads(match.strip())
                    if parsed_json:
                        return parsed_json
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON-like content without code blocks
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                cleaned_match = re.sub(r'\s+', ' ', match.strip())
                return json.loads(cleaned_match)
            except json.JSONDecodeError:
                continue
        
        # If all else fails, try to convert markdown to JSON
        try:
            return self.convert_markdown_to_json(response_text)
        except:
            pass
        
        return None

    def convert_markdown_to_json(self, markdown_text: str) -> Optional[dict]:
        """Convert the markdown response to JSON format."""
        import re
        
        # Extract explanation
        explanation_match = re.search(r'\*\*Contribution\*\*(.*?)(?=\*\*Nodes:\*\*)', markdown_text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "Urban planning contribution"
        
        # Extract nodes
        nodes = []
        node_pattern = r'\d+\.\s+\*\*(.*?)\*\*(.*?)(?=\d+\.\s+\*\*|\*\*Relationships:\*\*|$)'
        node_matches = re.findall(node_pattern, markdown_text, re.DOTALL)
        
        for node_name, node_content in node_matches:
            node_name = node_name.strip()
        
            # Extract labels
            labels_match = re.search(r'Labels:\s*"([^"]*)",?\s*"?([^"]*)"?', node_content)
            labels = []
            if labels_match:
                labels = [labels_match.group(1)]
                if labels_match.group(2):
                    labels.append(labels_match.group(2))
        
            # Extract properties
            description_match = re.search(r'Description:\s*(.*?)(?=\+|$)', node_content, re.DOTALL)
            description = description_match.group(1).strip() if description_match else ""
        
            node = {
                "name": node_name,
                "labels": labels or ["UrbanPlanning"],
                "properties": {
                    "description": description,
                    "domain": "Sustainable Urban Development",
                    "expertise_area": "Urban Planning & Design",
                    "analysis": f"Analysis for {node_name}",
                    "practical_applications": f"Practical applications for {node_name}"
                }
            }
            nodes.append(node)
        
        # Extract relationships
        relationships = []
        rel_pattern = r'\d+\.\s+\*\*(.*?)\s*->\s*(.*?)\*\*(.*?)(?=\d+\.\s+\*\*|\*\*Future Directions:\*\*|$)'
        rel_matches = re.findall(rel_pattern, markdown_text, re.DOTALL)
        
        for source, target, rel_content in rel_matches:
            relationship = {
                "source_node_name": source.strip(),
                "target_node_name": target.strip(),
                "type": "INFLUENCES",
                "properties": {
                    "tag": "causal",
                    "weight": 0.8,
                    "analysis": f"Relationship between {source.strip()} and {target.strip()}",
                    "cross_domain": True,
                    "evidence": "Urban planning research"
                }
            }
            relationships.append(relationship)
        
        return {
            "explanation": explanation,
            "nodes": nodes,
            "relationships": relationships,
            "future_directions": ["Explore cross-domain connections", "Investigate causal relationships"]
        }

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
                print(f"\nDEBUG: Raw response from {participant['name']}:\n{response_text[:500]}...\n")
                llm_response_json = self.extract_json_from_response(response_text)
                if not llm_response_json:
                    print(f"WARNING: Could not decode JSON from Anthropic for {participant['name']}")
                    print(f"Full response: {response_text}")
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
                    print(f"\nDEBUG: Raw response from {participant['name']}:\n{response_text[:500]}...\n")
                    llm_response_json = self.extract_json_from_response(response_text)
                    if not llm_response_json:
                        print(f"WARNING: Could not decode JSON from Ollama for {participant['name']}")
                        print(f"Full response: {response_text}")
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
            time.sleep(1)  # Be gentle on the LLMs and server

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
    # For demo purposes, we'll use the new setup
    game = KnowledgeGraphGame()
    
    # Set up the sustainable urban development participants
    game.setup_sustainable_urban_development_participants()
    
    # Initialize with the correct domain
    asyncio.run(game.initialize_graph(domain="sustainable_urban_development"))
    
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
