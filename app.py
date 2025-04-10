from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write("Made by Abrar")

def main():
    st.header("Chat with PDF")
    
    load_dotenv()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    
    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        text = ''
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Read all the pages and display the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        # st.write(chunks)
        
        
        #embeddings
        # embeddings=OpenAIEmbeddings()
        # VectorStore=FAISS.from_text(chunks ,embedding=embeddings)
        store_name=pdf.name[:-4]
        
        if os.path.exists("f{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
            st.write("Embeddings Loaded from the Disk")
        else:
            #embeddings
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks ,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
                
            # st.write("Embeddings Computation Completed")
            #Accept user  questions/query
            query=st.text_input("Ask any Question related to Your PDF file:")
            # st.write(query)
            if query:
                docs=VectorStore.similarity_search(query=query,k=1)
                # st.write(docs)
                llm=OpenAI()
                chain=load_qa_chain(llm=llm,chain_type='stuff')
                with get_openai_callback() as cb:
                    response=chain.run(input_documents=docs,question=query)
                    print(cb)
                st.write(response)

if __name__ == '__main__':
    main()
