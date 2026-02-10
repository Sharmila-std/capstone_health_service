import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Test with langchain_google_genai directly
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    print("Testing different Gemini model names...")
    print("-" * 50)
    
    # Try different model names
    models_to_try = [
        "models/gemini-pro",
        "gemini-pro", 
        "models/gemini-1.5-pro",
        "gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "gemini-1.5-flash"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\nTrying: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GEMINI_API_KEY,
                convert_system_message_to_human=True
            )
            response = llm.invoke("Say 'test successful'")
            print(f"✓ SUCCESS with {model_name}")
            print(f"  Response: {response.content[:50]}")
            break
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:100]}")
            
except Exception as e:
    print(f"Error: {e}")
