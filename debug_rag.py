import os
import sys
import traceback

print("--- Starting RAG Diagnostic ---")

# 1. Test Key
print(f"Checking GEMINI_API_KEY...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try looking in .env file directly if not in env vars yet
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        print(f"Loaded .env. Key found: {bool(api_key)}")
    except ImportError:
        print("python-dotenv not installed.")

if not api_key:
    print("[CRITICAL] GEMINI_API_KEY not found!")
else:
    print("[OK] GEMINI_API_KEY found.")

# 2. Test Imports
print("\nTesting Imports...")
try:
    import PyPDF2
    print("[OK] PyPDF2 imported")
except ImportError:
    print("[MISSING] PyPDF2 MISSING! Run: pip install PyPDF2")

try:
    from langchain_community.vectorstores import FAISS
    print("[OK] langchain_community.vectorstores FAISS imported")
except ImportError as e:
    print(f"[FAIL] langchain_community FAISS import failed: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("[OK] langchain_text_splitters imported")
except ImportError as e:
    print(f"[FAIL] langchain_text_splitters import failed: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("[OK] langchain_community HuggingFaceEmbeddings imported")
except ImportError as e:
    print(f"[FAIL] langchain_community HuggingFaceEmbeddings import failed: {e}")


# 3. Test Local Embedding Import
print("\nTesting Local Embedding Model (all-MiniLM-L6-v2)...")
try:
    # This triggers model download if not cached
    print("[WAIT] Loading model... (this may take time on first run)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    res = embeddings.embed_query("Hello test")
    print(f"[OK] Local Embedding success. Vector length: {len(res)}")
    
    # 4. Test FAISS Creation
    print("\nTesting FAISS Index Creation...")
    try:
        vector_store = FAISS.from_texts(["Hello test"], embedding=embeddings)
        print("[OK] FAISS Index created successfully.")
    except Exception as e:
        print(f"[FAIL] FAISS Creation failed: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"[FAIL] Local Embedding failed: {e}")
    traceback.print_exc()

print("\n--- Diagnostic Complete ---")
