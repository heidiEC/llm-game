import asyncio
import json
import time
from typing import Dict, List, Any
from game import KnowledgeGraphGame

class PlanningConsultationABTest:
    def __init__(self, knowledge_pod_client, llm_client):
        self.pod_client = knowledge_pod_client
        self.llm_client = llm_client
        self.test_results = []
        
    async def run_comparison_demo(self, user_question, town_profile):
        """Run side-by-side comparison of pod vs. no-pod responses"""
        
        print("\n" + "🔬 PLANNING CONSULTATION A/B TEST".center(80, "="))
        print(f"Question: {user_question}")
        print(f"Town: {town_profile['name']} ({town_profile['size']})")
        print("="*80)
        
        # Generate both responses
        print("⏳ Generating responses... (this may take a moment)\n")
        
        # Response A: LLM Only
        print("🤖 Generating LLM-only response...")
        llm_only_response = await self._generate_llm_only_response(user_question, town_profile)
        
        # Response B: Pod-Enhanced
        print("🧠 Generating Pod-enhanced response...")
        pod_enhanced_response = await self._generate_pod_enhanced_response(user_question, town_profile)
        
        # Display comparison
        self._display_comparison(llm_only_response, pod_enhanced_response, user_question)
        
        # Collect user feedback
        user_preference = self._collect_user_feedback()
        
        # Store results
        self._store_test_results(user_question, llm_only_response, pod_enhanced_response, user_preference)
        
        return {
            "llm_only": llm_only_response,
            "pod_enhanced": pod_enhanced_response,
            "user_preference": user_preference
        }
    
    async def _generate_llm_only_response(self, question, town_profile):
        """Generate response using LLM without knowledge pod"""
        prompt = f"""
        You are an expert urban planning consultant. Provide specific, actionable advice.
        
        Town Profile: {json.dumps(town_profile, indent=2)}
        
        Question: {question}
        
        Provide a comprehensive response with specific recommendations, considerations, and implementation steps.
        Keep your response practical and actionable for local planners.
        """
        
        if self.llm_client:
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
        else:
            response_text = "LLM client not available for comparison."
        
        return {
            "type": "LLM Only",
            "response": response_text,
            "knowledge_sources": "General training data",
            "specificity_score": self._calculate_specificity(response_text),
            "causal_relationships": self._count_causal_mentions(response_text),
            "population_mentions": self._count_population_mentions(response_text)
        }
    
    async def _generate_pod_enhanced_response(self, question, town_profile):
        """Generate response using LLM + knowledge pod"""
        
        # Get context from knowledge pod
        context_query = f"{question} {town_profile['vision']} {town_profile['geography']} {' '.join(town_profile['priorities'])}"
        context = await self.pod_client.get_context_for_query(context_query)
        
        # Extract specific insights
        causal_relationships = self._extract_causal_chains(context.get('context', []))
        population_impacts = self._extract_population_impacts(context.get('context', []))
        expert_perspectives = self._extract_expert_perspectives(context.get('context', []))
        
        prompt = f"""
        You are an expert urban planning consultant with access to a collaborative knowledge base 
        built by multiple domain experts. Use this knowledge to provide specific, evidence-based advice.
        
        Town Profile: {json.dumps(town_profile, indent=2)}
        
        Question: {question}
        
        Expert Knowledge Available:
        
        CAUSAL RELATIONSHIPS (with weights):
        {causal_relationships}
        
        POPULATION-SPECIFIC IMPACTS:
        {population_impacts}
        
        EXPERT PERSPECTIVES:
        {expert_perspectives}
        
        Instructions:
        1. Reference specific causal relationships with their weights (e.g., "0.8 causal weight")
        2. Mention trade-offs and conflicts identified by experts
        3. Differentiate impacts on specific population groups
        4. Cite which expert domain provided each insight (e.g., "Our Environmental Systems expert noted...")
        5. Provide concrete, measurable recommendations
        6. Explain WHY recommendations work based on expert knowledge
        
        Format your response to clearly show how the expert knowledge informs your recommendations.
        Use phrases like "Based on expert analysis..." and "Our knowledge base shows..."
        """
        
        if self.llm_client:
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
        else:
            response_text = "LLM client not available for comparison."
        
        return {
            "type": "Pod-Enhanced",
            "response": response_text,
            "knowledge_sources": f"{len(context.get('context', []))} expert-validated concepts",
            "specificity_score": self._calculate_specificity(response_text),
            "causal_relationships": self._count_causal_mentions(response_text),
            "population_mentions": self._count_population_mentions(response_text),
            "expert_citations": self._count_expert_citations(response_text),
            "context_nodes": len(context.get('context', []))
        }
    
    def _extract_causal_chains(self, context_nodes):
        """Extract causal relationships from context"""
        causal_info = []
        for node in context_nodes[:10]:  # Limit for readability
            props = node.get('properties', {})
            name = props.get('name', 'Unknown')
            analysis = props.get('analysis', '')
            
            if 'causal' in analysis.lower() or 'weight' in str(props):
                causal_info.append(f"- {name}: {analysis}")
        
        return "\n".join(causal_info) if causal_info else "No specific causal relationships identified."
    
    def _extract_population_impacts(self, context_nodes):
        """Extract population-specific impacts"""
        population_info = []
        population_keywords = ['families', 'residents', 'elderly', 'students', 'workers', 'income', 'demographic']
        
        for node in context_nodes:
            props = node.get('properties', {})
            name = props.get('name', '')
            description = props.get('description', '')
            
            if any(keyword in name.lower() or keyword in description.lower() for keyword in population_keywords):
                population_info.append(f"- {name}: {description}")
        
        return "\n".join(population_info[:8]) if population_info else "No specific population impacts identified."
    
    def _extract_expert_perspectives(self, context_nodes):
        """Extract expert domain perspectives"""
        expert_info = []
        for node in context_nodes[:8]:
            props = node.get('properties', {})
            expertise_area = props.get('expertise_area', 'General')
            name = props.get('name', 'Unknown')
            analysis = props.get('analysis', '')
            
            if expertise_area != 'General':
                expert_info.append(f"- {expertise_area}: {name} - {analysis[:100]}...")
        
        return "\n".join(expert_info) if expert_info else "General expert knowledge available."
    
    def _display_comparison(self, llm_response, pod_response, question):
        """Display side-by-side comparison"""
        
        print("\n" + "🤖 RESPONSE A: LLM ONLY".center(80, "="))
        print(f"Knowledge Source: {llm_response['knowledge_sources']}")
        print(f"Specificity Score: {llm_response['specificity_score']}/10")
        print(f"Causal Mentions: {llm_response['causal_relationships']}")
        print(f"Population Mentions: {llm_response['population_mentions']}")
        print("-" * 80)
        print(llm_response['response'])
        
        print("\n" + "🧠 RESPONSE B: POD-ENHANCED".center(80, "="))
        print(f"Knowledge Source: {pod_response['knowledge_sources']}")
        print(f"Specificity Score: {pod_response['specificity_score']}/10")
        print(f"Causal Mentions: {pod_response['causal_relationships']}")
        print(f"Population Mentions: {pod_response['population_mentions']}")
        print(f"Expert Citations: {pod_response.get('expert_citations', 0)}")
        print(f"Context Nodes Used: {pod_response.get('context_nodes', 0)}")
        print("-" * 80)
        print(pod_response['response'])
        
        print("\n" + "📊 QUICK COMPARISON".center(80, "="))
        print(f"Specificity:      A={llm_response['specificity_score']:2d} vs B={pod_response['specificity_score']:2d}")
        print(f"Causal Analysis:  A={llm_response['causal_relationships']:2d} vs B={pod_response['causal_relationships']:2d}")
        print(f"Population Focus: A={llm_response['population_mentions']:2d} vs B={pod_response['population_mentions']:2d}")
        print(f"Expert Validation: A= 0 vs B={pod_response.get('expert_citations', 0):2d}")
        
    def _collect_user_feedback(self):
        """Get user preference and reasoning"""
        print("\n" + "🗳️ YOUR FEEDBACK".center(80, "="))
        print("Which response would be more helpful for actual urban planning?")
        print("A) LLM Only - Uses general training knowledge")
        print("B) Pod-Enhanced - Uses expert-validated knowledge base") 
        print("C) Both equally helpful")
        print("D) Neither particularly helpful")
        
        choice = input("\nYour choice (A/B/C/D): ").upper()
        
        print("\nWhat made the difference for you?")
        reasoning = input("Brief explanation: ")
        
        return {
            "preference": choice,
            "reasoning": reasoning
        }
    
    def _store_test_results(self, question, llm_response, pod_response, user_feedback):
        """Store test results for analysis"""
        self.test_results.append({
            "question": question,
            "llm_metrics": {
                "specificity": llm_response['specificity_score'],
                "causal_mentions": llm_response['causal_relationships'],
                "population_mentions": llm_response['population_mentions']
            },
            "pod_metrics": {
                "specificity": pod_response['specificity_score'],
                "causal_mentions": pod_response['causal_relationships'],
                "population_mentions": pod_response['population_mentions'],
                "expert_citations": pod_response.get('expert_citations', 0),
                "context_nodes": pod_response.get('context_nodes', 0)
            },
            "user_feedback": user_feedback,
            "timestamp": time.time()
        })
    
    def _calculate_specificity(self, text):
        """Simple specificity scoring based on concrete terms"""
        specific_indicators = [
            'percent', '%', 'square feet', 'units', 'dollars', '$', 'timeline',
            'phase', 'metric', 'measure', 'specific', 'exactly', 'precisely',
            'zoning', 'ordinance', 'policy', 'regulation', 'standard', 'requirement',
            'minimum', 'maximum', 'target', 'goal', 'benchmark'
        ]
        
        score = sum(1 for indicator in specific_indicators if indicator.lower() in text.lower())
        return min(score, 10)  # Cap at 10
    
    def _count_causal_mentions(self, text):
        """Count causal relationship indicators"""
        causal_indicators = [
            'causes', 'leads to', 'results in', 'impacts', 'affects', 'influences',
            'because', 'due to', 'therefore', 'consequently', 'weight', 'relationship',
            'correlation', 'effect', 'outcome', 'drives', 'enables', 'prevents'
        ]
        
        return sum(1 for indicator in causal_indicators if indicator.lower() in text.lower())
    
    def _count_population_mentions(self, text):
        """Count population-specific mentions"""
        population_indicators = [
            'families', 'residents', 'elderly', 'seniors', 'students', 'workers',
            'children', 'youth', 'adults', 'income', 'demographic', 'community',
            'neighborhood', 'low-income', 'working class', 'professionals'
        ]
        
        return sum(1 for indicator in population_indicators if indicator.lower() in text.lower())
    
    def _count_expert_citations(self, text):
        """Count references to expert domains"""
        expert_domains = [
            'expert', 'specialist', 'analyst', 'environmental', 'social policy',
            'transportation', 'economic', 'community', 'climate', 'urban planning',
            'knowledge base', 'research shows', 'studies indicate', 'analysis reveals'
        ]
        
        return sum(1 for domain in expert_domains if domain.lower() in text.lower())

class InteractiveTownBuilder:
    def __init__(self, knowledge_pod_client, llm_client):
        self.pod_client = knowledge_pod_client
        self.llm_client = llm_client
        self.town_profile = {}
        self.conversation_history = []
        self.ab_tester = PlanningConsultationABTest(knowledge_pod_client, llm_client)
        
    async def start_consultation(self):
        """Begin the interactive town building process"""
        print("🏗️ " + "Welcome to the AI Urban Planning Consultation!".center(80, "="))
        print("I'm your AI planning consultant, backed by expert knowledge from multiple domains.")
        print("Let's design your ideal town together!")
        print("="*80)
        
        # Step 1: Gather basic town vision
        await self._gather_town_vision()
        
        # Step 2: Interactive planning conversation
        await self._interactive_planning_session()
        
        # Step 3: Generate final comprehensive plan
        await self._generate_final_plan()
        
    async def _gather_town_vision(self):
        """Collect initial town parameters from user"""
        print("\n📋 First, let's establish your town's basic profile:")
        
        # Town basics
        self.town_profile["name"] = input("\n🏘️  What would you like to name your town? ")
        
        print(f"\n🌍 What size town are you envisioning for {self.town_profile['name']}?")
        print("1. Small town (5,000-20,000 residents)")
        print("2. Medium town (20,000-100,000 residents)")  
        print("3. Small city (100,000-300,000 residents)")
        size_choice = input("Choose (1-3): ")
        
        size_map = {
            "1": "Small town (5,000-20,000 residents)",
            "2": "Medium town (20,000-100,000 residents)",
            "3": "Small city (100,000-300,000 residents)"
        }
        self.town_profile["size"] = size_map.get(size_choice, "Medium town")
        
        # Geographic context
        print(f"\n🗺️  What's the geographic setting for {self.town_profile['name']}?")
        geography = input("(e.g., 'coastal plains', 'mountain valley', 'river delta', 'prairie'): ")
        self.town_profile["geography"] = geography
        
        # Primary vision/goals
        print(f"\n🎯 What's your primary vision for {self.town_profile['name']}?")
        vision = input("(e.g., 'sustainable eco-town', 'tech hub', 'retirement community', 'college town'): ")
        self.town_profile["vision"] = vision
        
        # Key priorities
        print("\n📊 What are your top 3 priorities? (separate with commas)")
        print("Examples: affordable housing, environmental sustainability, economic growth, walkability, cultural preservation")
        priorities = input("Your priorities: ").split(",")
        self.town_profile["priorities"] = [p.strip() for p in priorities]
        
        print(f"\n✅ Great! Here's your town profile:")
        print("-" * 50)
        for key, value in self.town_profile.items():
            print(f"{key.title()}: {value}")
        print("-" * 50)
        
    async def _interactive_planning_session(self):
        """Main interactive consultation loop"""
        print(f"\n🤝 Now let's dive into planning {self.town_profile['name']}!")
        print("Ask me anything about urban planning, or I can suggest areas to explore.")
        print("💡 TIP: Type 'compare' to see side-by-side LLM vs Pod-Enhanced responses!")
        print("Type 'done' when you're ready for the final comprehensive plan.")
        
        # Get initial context from knowledge pod
        initial_context = await self._get_relevant_context(
            f"{self.town_profile['vision']} in {self.town_profile['geography']} focusing on {', '.join(self.town_profile['priorities'])}"
        )
        
        print(f"\n💡 Based on your vision, here are some key areas we should consider:")
        self._display_relevant_concepts(initial_context)
        
        while True:
            user_input = input("\n🗣️  Your question or area of interest: ").strip()
            
            if user_input.lower() in ['done', 'finished', 'complete']:
                break
                
            if user_input.lower() == 'compare':
                await self._run_comparison_mode()
                continue
                
            if user_input.lower() in ['help', 'suggestions']:
                await self._provide_suggestions()
                continue
            
            # Ask if they want comparison for this question
            want_comparison = input("🔬 Want to see LLM vs Pod comparison for this question? (y/n): ").lower() == 'y'
            
            if want_comparison:
                await self.ab_tester.run_comparison_demo(user_input, self.town_profile)
            else:
                # Regular pod-enhanced response
                response = await self._consult_on_topic(user_input)
                print(f"\n🏛️  Planning Consultant: {response}\n")
            
            self.conversation_history.append({
                "user_input": user_input,
                "had_comparison": want_comparison
            })
    
    async def _run_comparison_mode(self):
        """Dedicated comparison mode"""
        print("\n🔬 COMPARISON MODE ACTIVATED")
        print("Ask a planning question and see both LLM-only and Pod-enhanced responses!")
        
        question = input("🗣️  Your planning question: ")
        await self.ab_tester.run_comparison_demo(question, self.town_profile)
        
        print("\n✅ Comparison complete! Back to regular consultation...")
    
    async def _consult_on_topic(self, user_question):
        """Consult the knowledge pod and generate contextual advice"""
        
        # Get relevant context from knowledge pod
        context = await self._get_relevant_context(user_question)
        
        # Build rich prompt with pod knowledge
        prompt = f"""
        You are an expert urban planning consultant helping design a town. Use the provided expert knowledge to give specific, actionable advice.
        
        ## Town Profile
        {json.dumps(self.town_profile, indent=2)}
        
        ## User Question
        {user_question}
        
        ## Relevant Expert Knowledge
        {self._format_context_for_prompt(context)}
        
        ## Previous Conversation Context
        {self._format_conversation_history()}
        
        ## Instructions
        Provide specific, actionable advice that:
        1. Directly addresses the user's question
        2. References relevant concepts from the expert knowledge base
        3. Considers the town's specific profile and priorities
        4. Mentions potential trade-offs or considerations
        5. Suggests concrete next steps or follow-up questions
        
        Keep response conversational but informative (2-3 paragraphs max).
        """
        
        # Generate response using LLM with rich context
        if self.llm_client:
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            return "I'd love to help, but I need an LLM client to provide detailed consultation."
            
    async def _get_relevant_context(self, query):
        """Get relevant concepts from knowledge pod"""
        try:
            context = await self.pod_client.get_context_for_query(query)
            return context.get('context', [])
        except Exception as e:
            print(f"Warning: Could not access knowledge pod: {e}")
            return []
            
    def _display_relevant_concepts(self, context_nodes):
        """Display key concepts from knowledge pod"""
        if not context_nodes:
            print("  No specific concepts identified yet.")
            return
            
        # Group by domain/expertise
        concepts_by_domain = {}
        for node in context_nodes[:12]:  # Show top 12
            props = node.get('properties', {})
            domain = props.get('expertise_area', props.get('domain', 'General'))
            name = props.get('name', 'Unknown')
            
            if domain not in concepts_by_domain:
                concepts_by_domain[domain] = []
            concepts_by_domain[domain].append(name)
            
        for domain, concepts in concepts_by_domain.items():
            print(f"  🔹 {domain}: {', '.join(concepts[:3])}")
            if len(concepts) > 3:
                print(f"     ... and {len(concepts) - 3} more")
            
    async def _provide_suggestions(self):
        """Provide topic suggestions based on town profile"""
        suggestions = [
            "Housing strategy and affordability options",
            "Transportation and mobility planning", 
            "Environmental sustainability measures",
            "Economic development and job creation",
            "Community spaces and social infrastructure",
            "Climate resilience and adaptation",
            "Zoning and land use planning",
            "Infrastructure and utilities planning",
            "Public health and safety considerations",
            "Cultural preservation and community identity"
        ]
        
        print("\n💭 Here are some areas you might want to explore:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i:2d}. {suggestion}")
        print()
        
    def _format_context_for_prompt(self, context_nodes):
        """Format knowledge pod context for LLM prompt"""
        if not context_nodes:
            return "No specific expert knowledge available."
            
        formatted = "Key concepts and relationships from expert knowledge:\n"
        for node in context_nodes[:8]:  # Limit to avoid token overflow
            props = node.get('properties', {})
            name = props.get('name', 'Unknown')
            description = props.get('description', '')
            analysis = props.get('analysis', '')
            expertise = props.get('expertise_area', '')
            
            formatted += f"- {name}"
            if expertise:
                formatted += f" ({expertise})"
            formatted += f": {description}"
            if analysis:
                formatted += f" | Analysis: {analysis}"
            formatted += "\n"
            
        return formatted
        
    def _format_conversation_history(self):
        """Format previous conversation for context"""
        if not self.conversation_history:
            return "No previous conversation."
            
        formatted = "Previous discussion points:\n"
        for item in self.conversation_history[-3:]:  # Last 3 exchanges
            formatted += f"- User asked about: {item['user_input']}\n"
            
        return formatted
        
    async def _generate_final_plan(self):
        """Generate comprehensive final plan"""
        print("\n🎯 Generating your comprehensive town development plan...")
        print("This may take a moment as I synthesize all our discussions with expert knowledge...")
        
        # Get comprehensive context
        full_context_query = f"comprehensive urban planning for {self.town_profile['vision']} {self.town_profile['size']} in {self.town_profile['geography']}"
        context = await self._get_relevant_context(full_context_query)
        
        # Build comprehensive planning prompt
        prompt = f"""
        Create a comprehensive urban development plan based on our consultation.
        
        ## Town Profile
        {json.dumps(self.town_profile, indent=2)}
        
        ## Consultation Summary
        {self._format_conversation_history()}
        
        ## Expert Knowledge Base
        {self._format_context_for_prompt(context)}
        
        ## Plan Requirements
        Create a detailed 10-year development plan with:
        
        1. **Executive Summary** - Vision and key strategies
        2. **Housing Strategy** - Addressing affordability and community needs
        3. **Transportation Plan** - Mobility and connectivity
        4. **Environmental Framework** - Sustainability and resilience
        5. **Economic Development** - Job creation and local economy
        6. **Social Infrastructure** - Community spaces and services
        7. **Implementation Timeline** - Phased approach with milestones
        8. **Success Metrics** - How to measure progress
        
        Reference specific concepts from the expert knowledge base and explain causal relationships.
        Make it specific to {self.town_profile['name']} and actionable for local planners.
        """
        
        if self.llm_client:
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            plan = response.content[0].text
            
            print("\n" + "="*80)
            print(f"COMPREHENSIVE DEVELOPMENT PLAN FOR {self.town_profile['name'].upper()}")
            print("="*80)
            print(plan)
            print("\n" + "="*80)
            print("🎉 Your town planning consultation is complete!")
            print("Thank you for using the AI Urban Planning Assistant!")
            
            return plan
        else:
            print("Unable to generate final plan without LLM client.")
            return None

# Main demo runner
async def run_interactive_demo():
    """Run the full interactive demo"""
    
    print("🔧 " + "Initializing AI Urban Planning System".center(80, "="))
    print("Building expert knowledge base...")
    print("This will take a few minutes as our AI experts collaborate...")
    print("="*80)
    
    # 1. Build knowledge pod
    game = KnowledgeGraphGame()
    game.setup_sustainable_urban_development_participants()
    await game.initialize_graph(domain="sustainable_urban_development")
    
    # Quick knowledge building (fewer turns for demo)
    print(f"\n🤖 Running collaborative knowledge building with {len(game.participants)} experts...")
    results = await game.run_game(num_turns=12)  # 1.5 turns per expert
    
    successful_contributions = sum(1 for r in results if r and r.get('success'))
    print(f"\n✅ Knowledge base built! {successful_contributions}/{len(results)} successful expert contributions")
    
    # 2. Start interactive consultation
    print("\n🚀 Starting interactive consultation...")
    builder = InteractiveTownBuilder(
        knowledge_pod_client=game.mcp_client,
        llm_client=game.client
    )
    
    await builder.start_consultation()
    
    # 3. Show final statistics
    if builder.ab_tester.test_results:
        print("\n📈 A/B Test Results Summary:")
        print("="*50)
        for i, result in enumerate(builder.ab_tester.test_results, 1):
            print(f"Test {i}: {result['user_feedback']['preference']} - {result['user_feedback']['reasoning']}")

# Demo with sample questions
async def run_quick_demo():
    """Run a quick demo with pre-defined questions"""
    
    print("🚀 Quick Demo Mode - Testing with sample questions")
    
    # Build knowledge base
    game = KnowledgeGraphGame()
    game.setup_sustainable_urban_development_participants()
    await game.initialize_graph(domain="sustainable_urban_development")
    await game.run_game(num_turns=8)
    
    # Sample town profile
    sample_town = {
        "name": "Greenbrook Valley",
        "size": "Small town (5,000-20,000 residents)",
        "geography": "mountain valley with a river",
        "vision": "sustainable eco-town",
        "priorities": ["environmental sustainability", "walkability", "affordable housing"]
    }
    
    # Sample questions
    demo_questions = [
        "How should we handle affordable housing in our mountain valley eco-town?",
        "What's the best transportation strategy for a sustainable small town?",
        "How can we balance economic development with environmental protection?"
    ]
    
    ab_tester = PlanningConsultationABTest(game.mcp_client, game.client)
    
    for question in demo_questions:
        print(f"\n{'='*80}")
        print(f"DEMO QUESTION: {question}")
        await ab_tester.run_comparison_demo(question, sample_town)
        input("\nPress Enter to continue to next question...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_demo())
    else:
        asyncio.run(run_interactive_demo())