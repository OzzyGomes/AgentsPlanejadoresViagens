from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI

from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""

# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class TravelAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.OpenAIGPT4omini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        

    def expert_travel_agent(self):
        return Agent(
            role="Agente Especialista em viagens",
            backstory=dedent(f"""Especialista em planejamento e logística de viagens. Tenho décadas de experiência em fazer roteiros de viagens."""),
            goal=dedent(f"""Crie um itinerário de viagem de 7 dias com planos detalhados por dia,
                                incluem orçamento, sugestões de embalagem e dicas de segurança."""),
            tools=[SearchTools.search_internet,
                   CalculatorTools.calculate],
            verbose=True,
            llm=self.OpenAIGPT4omini,
        )

    def city_selection_expert(self):
        return Agent(
            role="Especialista em seleção de cidades",
            backstory=dedent(f"""Especialista em analisar dados de viagens para escolher destinos ideais"""),
            goal=dedent(f"""Selecione as melhores cidades com base no clima, estação do ano, preços e interesses dos viajantes"""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4omini,
        )
    
    def local_tour_guide(self):
        return Agent(
            role="Guia turístico local",
            backstory=dedent(f"""Guia local experiente com informações abrangentes
                                        sobre a cidade, seus atrativos e costumes"""),
            goal=dedent(f"""Forneça os MELHORES insights sobre a cidade selecionada"""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4omini,
        )