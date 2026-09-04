from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
import gradio as gr
from langchain_ollama import ChatOllama,OllamaEmbeddings



def get_llm():
    llama3 = ChatOllama(
        "llama3",
          temperature=0.2,
        num_predict=256,
    )
    return llama3

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document
def ollama_embedding():
    ollama_embedding = OllamaEmbeddings(
        model="nomic-embed-text:latest",  
        base_url="http://localhost:11434"  
    )
    return ollama_embedding


def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def vector_database(chunks):
    embedding_model = ollama_embedding()
    vectorDB = Chroma.from_documents(chunks,embedding_model)
    return vectorDB





def reteiever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectorDB = vector_database(chunks)
    reteiever = vectorDB.as_retriever()
    return reteiever

# End RAG SYSTEM

def retriever_qa(file,query):
    llm = get_llm()
    retriever_obj = reteiever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

rag_app = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"],type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
     outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    
)

rag_app.launch(server_name="0.0.0.0", server_port=7860, share=True)