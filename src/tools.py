from langchain.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from utils.scrap_pubmed_abstracts import scrap_pubmed
from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain import hub


def get_today_date(input : str) -> str:
    import datetime
    today = datetime.date.today()
    return f"\n {today} \n"

# Search
search = DuckDuckGoSearchResults()
# News
wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="d", max_results=5)
ddg_search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
# Wikipedia
wikipedia = WikipediaAPIWrapper()

def get_pubmed_paper_abstracts(input : str) -> str:
    df = scrap_pubmed(query=input, retmax=50)
    context_dict = {}
    for index in range(len(df)):
        context_dict[df.iloc[index]['Title']] = df.iloc[index]['Abstract']
    return f"\n {context_dict} \n"

get_pubmed_paper_abstracts_tool = Tool(
    name = "Get relevant pubmed papers",
    func = get_pubmed_paper_abstracts,
    description="Useful for scrapping relevant PubMed paper abstracts given the query to answer questions"
)

get_todays_date_tool = Tool(
    name="Get Todays Date",
    func=get_today_date,
    description="Useful for getting today's date"
)

get_duckduckgo_tool = Tool(
    name="DuckDuckGo",
    func=search.run,
    description="Useful for when you need to answer questions about current events"
)

get_wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Useful for when you need to answer basic questions that can be retrieved from Wikipedia"
)

