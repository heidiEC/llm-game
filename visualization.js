import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Lightbulb, Database, Network, Smile, Users } from 'lucide-react';

// A visualization component for our Knowledge Graph Game
export default function KnowledgeGraphVisualizer() {
  const [gameData, setGameData] = useState({
    graphStats: {
      nodeCount: 4,
      relationshipCount: 4,
      labelCount: 2,
      domainCount: 4
    },
    participants: [
      { name: "Claude-3-Opus", contributions: 0, color: "#8884d8" },
      { name: "Claude-3-Sonnet", contributions: 0, color: "#82ca9d" },
      { name: "GPT-4", contributions: 0, color: "#ffc658" }
    ],
    contributionTypes: [
      { name: "New Nodes", value: 0, color: "#8884d8" },
      { name: "New Relationships", value: 0, color: "#82ca9d" },
      { name: "New Labels", value: 0, color: "#ffc658" },
      { name: "Property Updates", value: 0, color: "#ff8042" }
    ],
    gameStatus: "Initialized with starting nodes",
    currentTurn: 0,
    maxTurns: 10
  });

  // Simulate game progress
  useEffect(() => {
    const timer = setTimeout(() => {
      if (gameData.currentTurn < gameData.maxTurns) {
        updateGameState();
      }
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [gameData]);

  const updateGameState = () => {
    // Simulate a turn in the game
    const newTurn = gameData.currentTurn + 1;
    
    // Generate a random update
    const participantIndex = Math.floor(Math.random() * gameData.participants.length);
    const contributionTypeIndex = Math.floor(Math.random() * gameData.contributionTypes.length);
    
    // Update stats
    const newGraphStats = {...gameData.graphStats};
    const newContributionTypes = [...gameData.contributionTypes];
    const newParticipants = [...gameData.participants];
    
    // Increase contribution count
    newContributionTypes[contributionTypeIndex].value += 1;
    newParticipants[participantIndex].contributions += 1;
    
    // Update graph stats based on contribution type
    switch(contributionTypeIndex) {
      case 0: // New Nodes
        newGraphStats.nodeCount += Math.floor(Math.random() * 3) + 1;
        newGraphStats.relationshipCount += Math.floor(Math.random() * 3) + 1;
        break;
      case 1: // New Relationships
        newGraphStats.relationshipCount += Math.floor(Math.random() * 3) + 1;
        break;
      case 2: // New Labels
        newGraphStats.labelCount += 1;
        break;
      case 3: // Property Updates
        // No stat change for property updates
        break;
      default:
        break;
    }
    
    // Game status update
    const participant = newParticipants[participantIndex];
    const contribution = newContributionTypes[contributionTypeIndex];
    const gameStatus = `Turn ${newTurn}: ${participant.name} added ${contribution.name}`;
    
    setGameData({
      ...gameData,
      graphStats: newGraphStats,
      participants: newParticipants,
      contributionTypes: newContributionTypes,
      gameStatus: gameStatus,
      currentTurn: newTurn
    });
  };

  return (
    <div className="flex flex-col p-6 max-w-full">
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold mb-2">LLM Knowledge Graph Game</h1>
        <p className="text-gray-600">A collaborative knowledge graph building experiment</p>
        <div className="mt-4 flex items-center justify-center">
          <div className="bg-blue-100 text-blue-800 px-4 py-2 rounded-lg flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <span>{gameData.gameStatus}</span>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Graph Stats Card */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Database className="mr-2" size={20} />
            Graph Statistics
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 p-3 rounded-md">
              <div className="text-sm text-gray-500">Nodes</div>
              <div className="text-2xl font-bold">{gameData.graphStats.nodeCount}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <div className="text-sm text-gray-500">Relationships</div>
              <div className="text-2xl font-bold">{gameData.graphStats.relationshipCount}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <div className="text-sm text-gray-500">Labels</div>
              <div className="text-2xl font-bold">{gameData.graphStats.labelCount}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <div className="text-sm text-gray-500">Domains</div>
              <div className="text-2xl font-bold">{gameData.graphStats.domainCount}</div>
            </div>
          </div>
        </div>
        
        {/* Game Progress Card */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Lightbulb className="mr-2" size={20} />
            Game Progress
          </h2>
          <div className="flex items-center mb-2">
            <div className="flex-1">
              <div className="text-sm text-gray-500 mb-1">Progress</div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: `${(gameData.currentTurn / gameData.maxTurns) * 100}%` }}
                ></div>
              </div>
            </div>
            <div className="ml-4 text-xl font-bold">
              {gameData.currentTurn}/{gameData.maxTurns}
            </div>
          </div>
          <div className="mt-4">
            <div className="text-sm text-gray-500 mb-2">Turn-by-Turn Activity</div>
            <div className="flex justify-between space-x-1">
              {Array.from({ length: gameData.maxTurns }).map((_, index) => (
                <div 
                  key={index}
                  className={`h-8 flex-1 rounded ${index < gameData.currentTurn ? 'bg-blue-500' : 'bg-gray-200'}`}
                ></div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Participant Contributions Chart */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Users className="mr-2" size={20} />
            Participant Contributions
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={gameData.participants}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="contributions" name="Contributions" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Contribution Types Chart */}
        <div className="bg-white p-4 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Network className="mr-2" size={20} />
            Contribution Types
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={gameData.contributionTypes}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Count" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Starting Knowledge */}
      <div className="mt-6 bg-white p-4 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Smile className="mr-2" size={20} />
          Initial Knowledge Graph
        </h2>
        <div className="bg-gray-50 p-4 rounded-md text-sm font-mono">
          <p>CREATE (biomimeticAlgo:Concept:ComputerScience {`{name: "Biomimetic Algorithms", description: "Algorithms inspired by biological processes in nature", domain: "Computer Science"}`})</p>
          <p>CREATE (antColony:Concept:Biology {`{name: "Ant Colony Optimization", description: "Swarm intelligence based on pheromone trail optimization", domain: "Biology"}`})</p>
          <p>CREATE (networkRouting:Concept:Telecommunications {`{name: "Network Routing Algorithms", description: "Methods for determining optimal paths in communication networks", domain: "Telecommunications"}`})</p>
          <p>CREATE (collectiveIntelligence:Concept:Behavior {`{name: "Collective Intelligence", description: "Shared or group intelligence emerging from collaboration", domain: "Complex Systems"}`})</p>
          <p>CREATE (biomimeticAlgo)-[:INSPIRED_BY]-&gt;(collectiveIntelligence)</p>
          <p>CREATE (antColony)-[:DEMONSTRATES]-&gt;(collectiveIntelligence)</p>
          <p>CREATE (biomimeticAlgo)-[:APPLIED_TO]-&gt;(networkRouting)</p>
          <p>CREATE (antColony)-[:SERVES_AS_MODEL_FOR]-&gt;(biomimeticAlgo)</p>
        </div>
      </div>
    </div>
  );
}

