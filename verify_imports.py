
try:
    import PyPDF2
    print("PyPDF2 imported successfully.")
except ImportError as e:
    print(f"Failed to import PyPDF2: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("FAISS imported successfully.")
except ImportError as e:
    print(f"Failed to import FAISS: {e}")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("LangChain Google GenAI imported successfully.")
except ImportError as e:
    print(f"Failed to import LangChain Google GenAI: {e}")
