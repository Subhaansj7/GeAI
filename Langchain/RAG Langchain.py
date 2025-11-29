from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

API_KEY='api key'

docs=[
    Document(page_content="LangChain is a framework for developing applications powered by LLMs."),
    Document(page_content="The core RAG components are Load, Split, Embed, Store, Retrieve, and Generate."),
    Document(page_content="Gemini is a family of multimodal models developed by Google."),
]

embeddngs= GoogleGenerativeAIEmbeddings(model='text-embedding-004',
                                        google_api_key=API_KEY)

vectorstore= Chroma.from_documents(docs,embeddngs)
retriever=vectorstore.as_retriever()


llm= ChatGoogleGenerativeAI(model='gemini-2.5-flash',
                            google_api_key=API_KEY)

rag_prompt= ChatPromptTemplate.from_template(
    "Answer the user's question ONLY based on the following context:\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_chain=(
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

question="what is the purpose of Langchain Framework?"
response=rag_chain.invoke(question)


print(response)
