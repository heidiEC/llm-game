LLM Knowledge Graph Game Rules
Overview
The LLM Knowledge Graph Game is a collaborative framework where multiple LLMs contribute to building a shared knowledge graph. Each LLM takes turns adding new elements to expand the graph in creative and insightful ways.
Game Structure
Initial Setup

Neo4j database serves as the knowledge graph backend
MCP (Model Context Protocol) facilitates communication between components
Pre-defined starting node(s) initialize the knowledge graph
Multiple LLMs are registered as participants

Turn-based Gameplay

Each turn, one LLM is selected to contribute
The LLM receives the current state of the knowledge graph
The LLM makes one type of contribution
The contribution is validated and added to the graph
The next LLM takes its turn

Contribution Types
LLMs can make ONE of the following types of contributions per turn:

Add Nodes (1-3 new nodes connected to existing nodes)

Must provide meaningful names and descriptions
Must create connections to at least one existing node


Add Relationships (new connections between existing nodes)

Must provide meaningful relationship types
Can add properties to describe relationship characteristics


Add Labels (taxonomic classification of nodes)

When patterns are recognized across nodes
Adds structure and categorization to the graph


Add/Modify Properties (enriching existing elements)

Can add or update properties of nodes or relationships
Properties must add meaningful information



Contribution Guidelines
Required Elements

Novelty: Information should be creative and insightful
Connectivity: Must connect to existing graph elements
Explanation: Brief explanation of what's being added and why it's interesting
Cypher Format: Provided in executable Neo4j Cypher syntax

Evaluation Criteria

Insight: Does the contribution reveal non-obvious connections?
Integration: How well does it connect with existing knowledge?
Accuracy: Is the information factually sound?
Creativity: Does it explore new dimensions of the topic?

Technical Implementation
Neo4j Graph Structure

Nodes represent concepts, entities, or ideas
Relationships represent connections between nodes
Labels provide categorical classification
Properties provide additional attributes

Cypher Commands
Contributions should be expressed as executable Cypher commands:
cypher// Creating nodes
CREATE (nodeID:Label {name: "Node Name", description: "Description"})

// Creating relationships
MATCH (a:Label {name: "Node A"}), (b:Label {name: "Node B"})
CREATE (a)-[:RELATIONSHIP_TYPE {property: "value"}]->(b)

// Adding labels
MATCH (n {name: "Node Name"})
SET n:NewLabel

// Adding/updating properties
MATCH (n {name: "Node Name"})
SET n.newProperty = "value"
Example Game Flow

Initialization:
Create starting node about "Biomimetic Algorithms" and related concepts
Turn 1 (LLM A):
Adds a node about "Slime Mold Networks" connected to both "Biomimetic Algorithms" and "Network Routing"
Turn 2 (LLM B):
Adds a new relationship between "Collective Intelligence" and "Network Routing" with type "EMERGES_FROM"
Turn 3 (LLM C):
Adds the label "NaturalOptimizer" to both "Ant Colony" and "Slime Mold Networks"
Turn 4 (LLM A):
Adds properties to "Biomimetic Algorithms" including "efficiency_gain" and "year_discovered"

Game Outcome
The collaborative process results in a rich knowledge graph that:

Reveals unexpected connections between domains
Builds a taxonomy of related concepts
Provides a structured representation of collective knowledge
Demonstrates the creative potential of collaborative LLM interaction