import os
from dotenv import load_dotenv
from rag_system import ConversationalRAGSystem

def main():
    # Load environment variables
    load_dotenv()
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("Error: Please set your GOOGLE_API_KEY in the .env file")
        return
    
    # Initialize RAG system
    print("🤖 Initializing RAG System...")
    rag_system = ConversationalRAGSystem(google_api_key)
    
    # Automatically load documents from data folder
    data_directory = "data"
    if os.path.exists(data_directory):
        print(f"📂 Loading documents from '{data_directory}' folder...")
        rag_system.load_documents_from_directory(data_directory)
        print("✅ Documents loaded successfully!")
    else:
        print("❌ Error: 'data' folder not found. Please create it and add your .txt files.")
        return
    
    # Start chat
    print("\n" + "="*50)
    print("🚀 RAG Chatbot Ready!")
    print("Ask questions about your documents!")
    print("Commands: 'clear' to clear history, 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\n💬 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        elif question.lower() == 'clear':
            rag_system.clear_conversation()
            continue
        
        elif not question:
            print("Please enter a question.")
            continue
        
        # Get response
        print("🤖 AI: ", end="")
        response = rag_system.ask_question(question)
        print(response)

if __name__ == "__main__":
    main()