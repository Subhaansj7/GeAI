from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage



chat= ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key='your_api_key'

)

messages=[
    HumanMessage(
        content="What are the 3 most important concepts of langchain framewrk?"
        )
]

response=chat.invoke(messages)

print(response.content)