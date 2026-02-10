import os
import sys
from rag_utils import get_embeddings
from langchain_community.vectorstores import FAISS

def verify_index(report_id):
    index_path = f"faiss_indexes/{report_id}"
    print(f"--- Verifying Index: {index_path} ---")
    
    if not os.path.exists(index_path):
        print("❌ Index folder not found!")
        return

    try:
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # Check number of documents (chunks)
        num_docs = vector_store.index.ntotal
        print(f"✅ Index loaded successfully.")
        print(f"📊 Total Vectors (Chunks): {num_docs}")
        
        if num_docs == 0:
            print("❌ Index is empty! No chunks stored.")
            return

        print("✅ FAISS Vector check passed.")
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_rag.py <report_id>")
        print("Explore 'faiss_indexes/' to find a report_id.")
    else:
        verify_index(sys.argv[1])
