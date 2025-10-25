from langchain_openai import ChatOpenAI
# Import the key using the name you prefer
from app.config import DEEPSEEK_API_KEY 

def get_llm():
    """
    Initializes and returns the ChatOpenAI client configured for OpenRouter,
    using the key stored under DEEPSEEK_API_KEY.
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found. Please check your .env file.")

    llm = ChatOpenAI(
        # The correct model ID you found
        model="tngtech/deepseek-r1t2-chimera:free", 
        # The OpenRouter endpoint
        base_url="https://openrouter.ai/api/v1",  
        # Pass the key value, regardless of the variable name
        api_key=DEEPSEEK_API_KEY              
    )
    print("✅ LLM Service initialized with OpenRouter (using DEEPSEEK_API_KEY var).")
    return llm

if __name__ == "__main__":
    # Test block to verify the LLM connection
    try:
        llm = get_llm()
        print("--- Testing LLM Service ---")
        response = llm.invoke("Hello, who are you?")
        print(f"LLM Response: {response.content}")
    except Exception as e:
        print(f"❌ Error testing LLM Service: {e}")