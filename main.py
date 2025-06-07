import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

print()

class PDFQueryAgent:
    def __init__(self, api_key):
        """Initialize the PDF Query Agent with Gemini API key"""
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.3,
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.retriever = None

    def load_pdf(self, pdf_path):
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            # persist_directory="./chroma_db"
        )

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 chunks
        )

        # Create QA chain
        self._create_qa_chain()
        
        print(f"Successfully loaded PDF: {pdf_path}")
        print(f"Created {len(chunks)} chunks")

    def _create_qa_chain(self):
        """Create the question-answering chain"""
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always mention which part of the document you're referencing when possible.

        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create the chain using LCEL
        self.qa_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question):
        """Query the PDF with a question"""
        if not self.qa_chain:
            return "Please load a PDF first using load_pdf() method."
        
        answer = self.qa_chain.invoke(question)
        return answer
    
    def query_with_sources(self, question):
        """Query the PDF and return answer with source references"""
        if not self.qa_chain or not self.retriever:
            return "Please load a PDF first using load_pdf() method."
        
        # Get answer
        answer = self.qa_chain.invoke(question)
        
        # Get source documents separately
        source_docs = self.retriever.get_relevant_documents(question) # use invoke instead of get_relevent_docs
        
        # Format response with sources
        response = f"Answer: {answer}\n\n"
        
        if source_docs:
            response += "Sources:\n"
            for i, doc in enumerate(source_docs, 1):
                page_num = doc.metadata.get('page', 'Unknown')
                source_name = doc.metadata.get('source', 'Unknown')
                response += f"\n{i}. Page {page_num}: {doc.page_content[:200]}...\n"
        
        return response
    
    def query_with_streaming(self, question):
        """Query with streaming response"""
        if not self.qa_chain:
            yield "Please load a PDF first using load_pdf() method."
            return
        
        for chunk in self.qa_chain.stream(question):
            yield chunk





if __name__ == "__main__":
    API_KEY = os.environ['GOOGLE_API_KEY']
    agent = PDFQueryAgent(API_KEY)

    # Load a PDF
    agent.load_pdf("Resume.pdf")

    # Query the PDF
    question = "What is the main topic of this document?"
    answer = agent.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Query with sources
    question = "What are the key findings?"
    response = agent.query_with_sources(question)
    print(f"\n{response}")