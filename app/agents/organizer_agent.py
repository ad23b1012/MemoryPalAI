# Import the LLM service we just built
from app.services.llm_service import get_llm 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

class OrganizerAgent:
    """
    An agent that analyzes text to extract entities and relationships
    to build a knowledge graph.
    """
    def __init__(self):
        # Get the LLM instance from our service
        self.llm = get_llm() 
        print("✅ OrganizerAgent initialized.")

    def extract_graph_data(self, text: str):
        """
        Uses an LLM to extract nodes (entities) and edges (relationships) from text.
        """
        print("\n🚀 Organizer Agent: Analyzing text to extract graph data...")

        # The parser will ensure the LLM's output is valid JSON.
        parser = JsonOutputParser()

        # A detailed prompt that instructs the LLM to return a JSON object
        # with 'nodes' (entities) and 'edges' (relationships).
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert at creating knowledge graphs from text.
            Analyze the following text and extract the key entities as nodes and the relationships between them as edges.

            RULES:
            - Nodes must have an 'id' (a unique name) and a 'type' (e.g., 'Person', 'Concept', 'Organization', 'Date', 'Location').
            - Edges must have a 'source' id, a 'target' id, and a 'label' describing their relationship (e.g., 'affiliated with', 'presented about', 'located in', 'mentioned on').
            - Only extract relationships explicitly mentioned or strongly implied in the text.

            TEXT:
            {text_input}

            {format_instructions}
            """
        )

        # Create the chain that links the prompt, LLM, and parser.
        chain = prompt | self.llm | parser

        try:
            graph_data = chain.invoke({
                "text_input": text,
                "format_instructions": parser.get_format_instructions(),
            })
            print("✅ Organizer Agent: Successfully extracted graph data.")
            return graph_data
        except Exception as e:
            print(f"❌ Organizer Agent: Failed to extract data. Error: {e}")
            return None

if __name__ == "__main__":
    # This block allows us to run this file directly to test it.
    print("--- Testing OrganizerAgent ---")
    
    organizer = OrganizerAgent()

    # Let's use the sample text from the PDF we tested earlier
    test_text = """
    Overview of AI and Generative AI
    Dr. Sunil Saumya
    Dept of Data Science & Artificial Intelligence
    Indian Institute of Information Technology Dharwad
    sunilsaumya @iiitdwd.ac.in
    August 6, 2025
    Dr. Sunil Saumya (IIIT Dharwad) Agentic AI August 6, 2025 1 / 32
    """

    knowledge_graph_data = organizer.extract_graph_data(test_text)

    if knowledge_graph_data:
        print("\n--- Extracted Knowledge Graph ---")
        # Pretty-print the JSON output
        print(json.dumps(knowledge_graph_data, indent=2))
        print("---------------------------------")