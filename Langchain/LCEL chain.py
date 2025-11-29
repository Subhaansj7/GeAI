from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key='your_apui_key'
)

output_parser= StrOutputParser()

prompt=ChatPromptTemplate.from_template(
    'You are a helpful assistant. Provide a simple, 2-point summary of the concept: {concept}'

)

chain= prompt | chat | output_parser

response=chain.invoke({'concept':'Recursion in programming'})


print(response)
