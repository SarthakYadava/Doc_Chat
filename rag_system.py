import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from memory_manager import ConversationMemory

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    conversation_history: str

class ConversationalRAGSystem:
    def __init__(self, google_api_key: str):
        # Initialize LLM with updated model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        # Initialize embeddings with updated model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        
        # Initialize vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=10)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context and conversation history.

Previous Conversation:
{conversation_history}

Context from documents:
{context}

Current Question: {question}

Instructions:
1. Use the provided context to answer the question accurately
2. Consider the conversation history to maintain context
3. If the question relates to previous conversation, reference it appropriately
4. If you cannot find the answer in the context, say so politely
5. Keep responses concise but informative
6. Be conversational and friendly

Answer:""")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def load_documents_from_directory(self, directory_path: str):
        """Load all text documents from directory"""
        try:
            documents = []
            
            # Check if directory exists
            if not os.path.exists(directory_path):
                print(f"Error: Directory '{directory_path}' not found")
                return []
            
            # Load all text files
            loaded_files = []
            for filename in os.listdir(directory_path):
                if filename.endswith(('.txt', '.md')):
                    file_path = os.path.join(directory_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            if content.strip():  # Only add non-empty files
                                doc = Document(
                                    page_content=content, 
                                    metadata={"source": filename}
                                )
                                documents.append(doc)
                                loaded_files.append(filename)
                    except Exception as e:
                        print(f"Error reading file {filename}: {str(e)}")
            
            if not documents:
                print("No valid text files found in the directory")
                return []
            
            print(f"Loaded files: {', '.join(loaded_files)}")
            return self._process_and_store_documents(documents)
        
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []
    
    def _process_and_store_documents(self, documents: List[Document]):
        """Process and store documents in vector store"""
        try:
            # Split documents into chunks
            all_splits = self.text_splitter.split_documents(documents)
            print(f"Split into {len(all_splits)} chunks")
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=all_splits)
            print(f"Added {len(document_ids)} chunks to vector store")
            
            return document_ids
        
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return []
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        
        def retrieve(state: State):
            # Get conversation context
            conversation_context = self.memory.get_conversation_context()
            
            # Enhanced query with conversation context
            enhanced_query = state["question"]
            if conversation_context:
                recent_topics = self.memory.get_recent_topics()
                if recent_topics:
                    enhanced_query += " " + " ".join(recent_topics[:3])
            
            # Retrieve documents
            retrieved_docs = self.vector_store.similarity_search(
                enhanced_query, 
                k=4
            )
            
            return {
                "context": retrieved_docs,
                "conversation_history": conversation_context
            }
        
        def generate(state: State):
            # Format context
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            
            # Create prompt
            formatted_prompt = self.prompt_template.format(
                conversation_history=state["conversation_history"],
                context=docs_content,
                question=state["question"]
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            return {"answer": response.content}
        
        # Build graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get response"""
        try:
            # Run the graph
            result = self.graph.invoke({"question": question})
            
            # Store in memory
            context_sources = [doc.metadata.get("source", "unknown") for doc in result["context"]]
            self.memory.add_exchange(
                user_input=question,
                ai_response=result["answer"],
                context=context_sources
            )
            
            return result["answer"]
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return list(self.memory.conversation_history)
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory.clear_history()
        print("Conversation history cleared!")