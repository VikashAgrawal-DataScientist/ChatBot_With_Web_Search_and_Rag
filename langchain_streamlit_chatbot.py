# streamlit_app.py
import streamlit as st
import tempfile
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
try:
    from langchain_tavily import TavilySearch
    TAVILY_NEW = True
except ImportError:
    st.warning("Install langchain-tavily: pip install langchain-tavily")
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_NEW = False

# Set up the page
st.set_page_config(page_title="Smart RAG Assistant")
st.title("Smart Assistant with RAG & Web Search")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password", 
                                  help="Get from https://tavily.com/ (optional)")
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", 
                                   type=["pdf", "txt"])
    
    # Document processing settings
    st.subheader("Processing Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    if st.button("Clear Document Memory"):
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
            st.success("Document memory cleared!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize vector store in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Function to process uploaded document
def process_document(file, chunk_size=1000, chunk_overlap=200):
    """Process uploaded document and create vector store"""
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # Load document based on file type
        if file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:  # text files
            loader = TextLoader(tmp_file_path)
        
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vector_store, len(chunks)
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, 0

# Process uploaded document
if uploaded_file is not None and openai_api_key.startswith('sk-'):
    with st.spinner("Processing document..."):
        vector_store, num_chunks = process_document(
            uploaded_file, chunk_size, chunk_overlap
        )
        if vector_store:
            st.session_state.vector_store = vector_store
            st.sidebar.success(f"✅ Document processed! Created {num_chunks} chunks.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to create RAG chain
def create_rag_chain(vector_store, openai_api_key):
    """Create a RAG chain with the given vector store"""
    
    # Custom prompt for RAG
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in a helpful and concise manner:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# Function for web search
def perform_web_search(query, tavily_api_key):
    """Perform web search using Tavily"""
    try:
        if TAVILY_NEW:
            # Use the new TavilySearch class
            search_tool = TavilySearch(tavily_api_key=tavily_api_key, max_results=3)
        else:
            # Use the deprecated TavilySearchResults with search_depth="basic" to avoid empty results
            search_tool = TavilySearchResults(
                tavily_api_key=tavily_api_key, 
                max_results=3,
                search_depth="basic"  # Use "basic" instead of "advanced" to avoid empty results
            )
        
        results = search_tool.invoke({"query": query})
        
        # Handle the results properly - they can be in different formats
        if isinstance(results, list):
            # If results are a list, extract content from each item
            formatted_results = []
            for item in results:
                if isinstance(item, dict):
                    # Extract relevant information from dictionary items
                    content = item.get('content', '') or item.get('snippet', '') or str(item)
                    formatted_results.append(content)
                else:
                    formatted_results.append(str(item))
            return "\n".join(formatted_results)
        elif isinstance(results, str):
            # If results are already a string, return as is
            return results
        else:
            # Fallback: convert to string
            return str(results)
            
    except Exception as e:
        return f"Web search unavailable: {str(e)}"

# Main chat interface
if prompt := st.chat_input("Ask a question..."):
    # Validate API key
    if not openai_api_key.startswith('sk-'):
        st.warning("Please enter a valid OpenAI API key to continue!")
        st.stop()
    
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        full_response = ""
        
        try:
            # Check if we have a document and should use RAG
            if st.session_state.vector_store is not None:
                # Use RAG for document-based questions
                rag_chain = create_rag_chain(st.session_state.vector_store, openai_api_key)
                response = rag_chain.invoke(prompt)
                
                full_response = response["result"]
                
                # Add source information
                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in response["source_documents"]]))
                if sources:
                    full_response += f"\n\n**Sources:** {', '.join(sources)}"
                    
            else:
                # Check if web search is available and query needs current information
                needs_search = any(keyword in prompt.lower() for keyword in 
                                 ["current", "recent", "latest", "today", "now", "news", "update", "202"])
                
                if needs_search and tavily_api_key:
                    # Try web search first
                    search_results = perform_web_search(prompt, tavily_api_key)
                    if search_results and not search_results.startswith("Web search unavailable"):
                        # Properly format the search results
                        enhanced_prompt = f"""Based on the following web search results and your knowledge, answer the question:

Search Results:
{search_results}

Question: {prompt}
Please provide a comprehensive answer based on the search results above:"""
                        
                        llm = ChatOpenAI(
                            openai_api_key=openai_api_key,
                            model_name="gpt-3.5-turbo",
                            temperature=0.7
                        )
                        response = llm.invoke(enhanced_prompt)
                        full_response = response.content
                        full_response += "\n\n*🔍 This answer includes information from web search.*"
                    else:
                        # Fallback to regular LLM
                        llm = ChatOpenAI(
                            openai_api_key=openai_api_key,
                            model_name="gpt-3.5-turbo",
                            temperature=0.7
                        )
                        response = llm.invoke(prompt)
                        full_response = response.content
                        if search_results.startswith("Web search unavailable"):
                            full_response += f"\n\n*⚠️ Note: {search_results}*"
                else:
                    # Regular LLM response
                    llm = ChatOpenAI(
                        openai_api_key=openai_api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=0.7
                    )
                    response = llm.invoke(prompt)
                    full_response = response.content
                    
                    if needs_search and not tavily_api_key:
                        full_response += "\n\n*💡 For current information, add a Tavily API key to enable web search.*"
            
            # Display the response
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display current status
with st.sidebar:
    st.subheader("Current Status")
    if st.session_state.vector_store is not None:
        st.success("✅ Document loaded and ready for questions")
    else:
        st.info("📝 No document loaded - using general knowledge")
    
    if tavily_api_key:
        st.success("🌐 Web search enabled")
    else:
        st.info("🌐 Web search disabled - add Tavily API key")
