
import os
import pickle
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_embeddings():
    # Use Local Embeddings (CPU) - Free & Unlimited
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_report(filepath, report_id):
    """
    Reads a PDF, chunks text, creates FAISS index, saves it locally.
    Returns: success (bool), message (str)
    """
    try:
        text = ""
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text()
        
        if not text:
            return False, "Could not extract text from PDF."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Save index locally
        index_path = f"faiss_indexes/{report_id}"
        os.makedirs("faiss_indexes", exist_ok=True)
        vector_store.save_local(index_path)
        
        return True, "Report processed successfully."
    except Exception as e:
        print(f"Error processing report: {e}")
        return False, str(e)

def ask_report(report_id, question, language="English", empathetic=False):
    """
    Asks a question to the report using vector search and minimal Gemini API usage.
    - For summaries: Returns extracted text directly from vectors (NO API call)
    - For specific questions: Sends only keywords to Gemini (minimal API usage)
    Returns: answer (str), source_docs (list)
    """
    try:
        # Load FAISS index
        index_path = f"faiss_indexes/{report_id}"
        if not os.path.exists(index_path):
            return "Error: Report index not found.", []
        
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        question_lower = question.lower()
        
        # Only summarize queries are extractive (no API call)
        # All other questions use Gemini API
        is_summary_query = any(word in question_lower for word in ['summarize', 'summary', 'overview'])
        
        # Get relevant context from the report
        if is_summary_query:
            # For summaries, get more chunks
            docs = vector_store.similarity_search(question, k=5)
        else:
            # For specific questions, get fewer chunks
            docs = vector_store.similarity_search(question, k=2)
        
        if not docs:
            return "I couldn't find relevant information in the report to answer this question.", []
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Decide if we need Gemini for summary (for translation or tone)
        needs_gemini_for_summary = (language != "English" or empathetic)
        
        # FOR SUMMARY QUERIES: 
        if is_summary_query and not needs_gemini_for_summary:
            # Standard English summary - return extracted text directly (Free)
            summary = f"""**Report Summary:**

{context[:1500]}

**Note:** This is extracted directly from your report. For specific questions or detailed analysis, please ask me directly.

Please consult your healthcare provider for proper interpretation."""
            return summary, docs
        
        # If we need translation/tone for summary, OR if it's a specific question, use Gemini
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Extract only key information to send to Gemini (reduce token usage)
                # Send max 500 characters instead of 3000
                minimal_context = context[:500]
                
                # Enhanced prompt with language and empathy
                style_instruction = ""
                if empathetic:
                    style_instruction = "Use a very kind, empathetic, and supportive tone, as if you are explaining this to a beloved family member. Avoid medical jargon; use simple, comforting words. "
                
                lang_instruction = f"IMPORTANT: provide the final answer ONLY in {language}." if language != "English" else ""

                prompt = f"""{style_instruction}Based on this medical report excerpt, answer briefly:

Report: {minimal_context}

Question: {question}

{lang_instruction}

Answer concisely. End with: "Consult your healthcare provider for guidance."

Answer:"""
                
                response = model.generate_content(prompt)
                answer = response.text
                
                return answer, docs
            except Exception as e:
                print(f"Gemini API error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to extractive answer
                return f"**Relevant Information:**\n\n{context[:600]}\n\n**Note:** Please consult with a healthcare professional.", docs
        else:
            # No API key - use extractive answer
            return f"**Relevant Information:**\n\n{context[:600]}\n\n**Note:** Please consult with a healthcare professional.", docs
        
    except Exception as e:
        print(f"Error asking report: {e}")
        import traceback
        traceback.print_exc()
        return f"I apologize, but I encountered an error processing your question. Please try rephrasing it.", []

def recommend_next_steps(answer_text):
    """
    Analyzes the answer to suggest ONLY actionable next steps:
    - Test center bookings for scans/diagnostics
    - Doctor appointments for specialists
    
    Returns dict with recommendation flags and types.
    """
    try:
        answer_lower = answer_text.lower()
        
        # Initialize recommendations as False
        recommend_scan = False
        scan_type = None
        recommend_doctor = False
        specialist_type = None
        
        # Check for specific scan/test mentions
        scan_keywords = {
            'mri': 'MRI Scan',
            'ct scan': 'CT Scan',
            'ct': 'CT Scan',
            'x-ray': 'X-Ray',
            'xray': 'X-Ray',
            'ultrasound': 'Ultrasound',
            'ecg': 'ECG/EKG',
            'ekg': 'ECG/EKG',
            'blood test': 'Blood Test',
            'urine test': 'Urine Test',
            'biopsy': 'Biopsy',
            'mammogram': 'Mammogram',
            'endoscopy': 'Endoscopy',
            'colonoscopy': 'Colonoscopy'
        }
        
        for keyword, scan_name in scan_keywords.items():
            if keyword in answer_lower:
                recommend_scan = True
                scan_type = scan_name
                break
        
        # Check for specific specialist mentions
        doctor_keywords = {
            'cardiologist': 'Cardiologist',
            'heart': 'Cardiologist',
            'cardiac': 'Cardiologist',
            'neurologist': 'Neurologist',
            'brain': 'Neurologist',
            'nerve': 'Neurologist',
            'dermatologist': 'Dermatologist',
            'skin': 'Dermatologist',
            'orthopedic': 'Orthopedic Specialist',
            'bone': 'Orthopedic Specialist',
            'joint': 'Orthopedic Specialist',
            'endocrinologist': 'Endocrinologist',
            'diabetes': 'Endocrinologist',
            'thyroid': 'Endocrinologist',
            'hormone': 'Endocrinologist',
            'gastroenterologist': 'Gastroenterologist',
            'stomach': 'Gastroenterologist',
            'digestive': 'Gastroenterologist',
            'pulmonologist': 'Pulmonologist',
            'lung': 'Pulmonologist',
            'respiratory': 'Pulmonologist',
            'nephrologist': 'Nephrologist',
            'kidney': 'Nephrologist',
            'urologist': 'Urologist',
            'oncologist': 'Oncologist',
            'cancer': 'Oncologist',
            'tumor': 'Oncologist',
            'ophthalmologist': 'Ophthalmologist',
            'eye': 'Ophthalmologist',
            'vision': 'Ophthalmologist',
            'ent': 'ENT Specialist',
            'ear': 'ENT Specialist',
            'nose': 'ENT Specialist',
            'throat': 'ENT Specialist'
        }
        
        for keyword, specialist in doctor_keywords.items():
            if keyword in answer_lower:
                recommend_doctor = True
                specialist_type = specialist
                break
        
        # Check for general medical consultation indicators
        consultation_keywords = ['consult', 'see a doctor', 'medical attention', 'healthcare provider', 'physician', 'abnormal', 'concern']
        if any(keyword in answer_lower for keyword in consultation_keywords) and not recommend_doctor:
            recommend_doctor = True
            specialist_type = 'General Physician'
        
        return {
            "recommend_scan": recommend_scan,
            "scan_type": scan_type,
            "recommend_doctor": recommend_doctor,
            "specialist_type": specialist_type
        }
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        # Return empty recommendations on error
        return {
            "recommend_scan": False,
            "scan_type": None,
            "recommend_doctor": False,
            "specialist_type": None
        }

def ask_general_question(report_id, question, language="English", empathetic=False):
    """
    Answers general questions about the report using Gemini API with MINIMAL context.
    Optimized to reduce API token usage.
    Returns: answer (str)
    """
    try:
        # Load FAISS index to get report context
        index_path = f"faiss_indexes/{report_id}"
        if not os.path.exists(index_path):
            return "Error: Report index not found."
        
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # Get minimal context for general questions (reduced from k=6 to k=3)
        docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return "I couldn't find relevant information in the report."
        
        # Combine context - REDUCED from 4000 to 500 characters
        context = "\n\n".join([doc.page_content for doc in docs])
        minimal_context = context[:500]
        
        # Use Gemini API with minimal context
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Enhanced prompt with language and empathy
                style_instruction = ""
                if empathetic:
                    style_instruction = "Explain this with extreme kindness and care. Use simple, non-technical language that is easy for a non-medical person to understand. Be supportive. "
                
                lang_instruction = f"IMPORTANT: provide the final answer ONLY in {language}." if language != "English" else ""

                prompt = f"""{style_instruction}Answer this question about the medical report briefly:

Report excerpt: {minimal_context}

Question: {question}

{lang_instruction}

Provide a concise, helpful answer. End with: "Consult your healthcare provider for guidance."

Answer:"""
                
                response = model.generate_content(prompt)
                answer = response.text
                
                return answer
            except Exception as e:
                print(f"Gemini API error: {e}")
                import traceback
                traceback.print_exc()
                return f"I apologize, but I encountered an error processing your question. Please try again or rephrase your question."
        else:
            return "Gemini API is not configured. Please contact support."
        
    except Exception as e:
        print(f"Error in general Q&A: {e}")
        import traceback
        traceback.print_exc()
        return f"I apologize, but I encountered an error. Please try rephrasing your question."
