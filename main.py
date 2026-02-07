from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langsmith import traceable
load_dotenv(override=True)
@traceable(project_name="hello-world-vibhakar")
def main():
    print("Hello from langchain-course-vibhakar!")
    information = "Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate."
    summaryPromptTemplate = "Taking inputs from the {information} create a summary of the information and provide 2 interesting facts about the person."
    
    summaryPropmtTemplateInputVariables = PromptTemplate(input_variables=["information"], template=summaryPromptTemplate)

    

    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)
    summaryChain = summaryPropmtTemplateInputVariables | llm 

    response=summaryChain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
