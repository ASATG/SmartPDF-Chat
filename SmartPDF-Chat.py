import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
import os
import together
from typing import Any, Dict
from pydantic import Extra
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import wordninja
from langchain.vectorstores import FAISS
import textwrap

def process(d):
    os.environ["TOGETHER_API_KEY"] = ""
    together.api_key = os.environ["TOGETHER_API_KEY"]

    class TogetherLLM(LLM):
        """Together large language models."""

        model: str = "togethercomputer/llama-2-70b-chat"
        """model endpoint to use"""

        together_api_key: str = os.environ["TOGETHER_API_KEY"]
        """Together API key"""

        temperature: float = 0.1
        """What sampling temperature to use."""

        max_tokens: int = 1024
        """The maximum number of tokens to generate in the completion."""

        class Config:
            extra = Extra.forbid

        def validate_environment(cls, values: Dict) -> Dict:
            """Validate that the API key is set."""
            api_key = get_from_dict_or_env(
                values, "together_api_key", "TOGETHER_API_KEY"
            )
            values["together_api_key"] = api_key
            return values

        def _llm_type(self) -> str:
            """Return type of LLM."""
            return "together"

        def _call(
            self,
            prompt: str,
            **kwargs: Any,
        ) -> str:
            """Call to Together endpoint."""
            together.api_key = self.together_api_key
            output = together.Complete.create(prompt,
                                            model=self.model,
                                            max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            )
            text = output['output']['choices'][0]['text']
            return text
    lc_embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    start_and_end=d

    mapping={}
    all_chunks=[]

    for key in start_and_end:
        start_page=start_and_end[key][0]
        end_page=start_and_end[key][1]

        for page_number in range(start_page - 1, end_page, 1):
            page = start_and_end[key][2].pages[page_number]
            temp_text=" ".join(wordninja.split(page.extract_text()))
            temp_chunks=RecursiveCharacterTextSplitter(
                    chunk_size=1024, chunk_overlap=250, length_function=len
                ).split_text(temp_text)
            mapping[key+"_"+str(page_number+1)]=temp_chunks
            all_chunks.extend(temp_chunks)

    vectordb = FAISS.from_texts(texts=all_chunks,embedding=lc_embed_model)

    llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature = 0.4,
    max_tokens = 1024
    )
    
    return vectordb, llm, mapping


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    return (wrap_text_preserve_newlines(llm_response['result']))


with st.sidebar:
    st.title('PDF ChatBot')
    st.markdown('''
                Made By: 
                - Ayush Ambhorkar(112103016)
                - Vipul Chaudhari(112103028)
                - Aditya Choudhary(112103030)
                '''
    )
    pdf_dict={}
    pdfs = st.file_uploader("Upload your PDFs",type='pdf',accept_multiple_files=True)
    i = 1
    for pdf in pdfs:
        n1 = st.number_input(pdf.name + " Start Page")
        n2 = st.number_input(pdf.name + " End Page")
        obj = [int(n1), int(n2), PdfReader(pdf)]
        pdf_dict[pdf.name] = obj
        st.write(pdf_dict[pdf.name])
        i+=1
        
    done = st.button('DONE')
    

st.header("Your Personalized PDF Chatbot")
    
if done:
    vectordb, llm, mapping = process(pdf_dict)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True)
    st.session_state.vectordb = vectordb
    st.session_state.mapping = mapping
    st.session_state.llm = llm
    st.session_state.retriever = retriever
    st.session_state.qa_chain = qa_chain
   
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if base_query := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(base_query)
    st.session_state.messages.append({"role": "user", "content": base_query})
    backupans=st.session_state.vectordb.similarity_search_with_relevance_scores(query=base_query, k=10,score_threshold=0.5)
    count = 0
    for item in backupans:
        for page in st.session_state.mapping:
            if str(item[0])[14:-1] in st.session_state.mapping[page]:
                # print(item[1],page,'->',str(item[0])[14:-1],end='\n\n')
                if(item[1]) > 0.5:
                    count += 1
    if count > 0:
        llm_response = st.session_state.qa_chain(base_query)
        ans = process_llm_response(llm_response)
    else:
        ans = "The answer of the given query is not in the context!!!"
    response = f"{ans}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    