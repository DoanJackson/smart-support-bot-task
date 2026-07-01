import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

SYSTEM_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.
• Tone: helpful, factual, concise.
• Only answer using the uploaded docs.
• Max 5 bullet points; else link to the doc.
• Cite up to 3 "Article URL:" lines per reply."""

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Error: GEMINI_API_KEY is not set in .env")
        return

    # Load vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
    persist_directory = "data/chroma_db"
    
    if not os.path.exists(persist_directory):
        print(f"Error: {persist_directory} not found. Please run vector_store_setup.py first.")
        return
        
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize Gemini LLM - added "models/" prefix to fix NOT_FOUND error
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash", temperature=0, google_api_key=api_key)

    # Create Prompt Template
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )

    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\nOptiBot Assistant is ready! Type 'exit' or 'quit' to stop.\n" + "="*50)
    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        if not question.strip():
            continue
            
        print("-" * 50)
        response = retrieval_chain.invoke({"input": question})
        
        print("\nOptiBot:")
        print(response['answer'])
        
        print("\n[Sources Retrieved]:")
        
        # Track unique sources to avoid duplicate printing
        unique_sources = set()
        for doc in response['context']:
            source = doc.metadata.get('source', 'Unknown')
            unique_sources.add(source)
            
        for i, source in enumerate(unique_sources):
            print(f"  {i+1}. {source}")
        print("-" * 50)

if __name__ == "__main__":
    main()
