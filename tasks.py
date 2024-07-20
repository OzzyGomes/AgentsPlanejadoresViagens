from crewai import Task
from textwrap import dedent


"""
Creating Tasks Cheat Sheet:
- Begin with the end in mind. Identify the specific outcome your tasks are aiming to achieve.
- Break down the outcome into actionable tasks, assigning each task to the appropriate agent.
- Ensure tasks are descriptive, providing clear instructions and expected deliverables.

Goal:
- Develop a detailed itinerary, including city selection, attractions, and practical travel advice.

Key Steps for Task Creation:
1. Identify the Desired Outcome: Define what success looks like for your project.
    - A detailed 7 day travel itenerary.

2. Task Breakdown: Divide the goal into smaller, manageable tasks that agents can execute.
    - Itenerary Planning: develop a detailed plan for each day of the trip.
    - City Selection: Analayze and pick the best cities to visit.
    - Local Tour Guide: Find a local expert to provide insights and recommendations.

3. Assign Tasks to Agents: Match tasks with agents based on their roles and expertise.

4. Task Description Template:
  - Use this template as a guide to define each task in your CrewAI application. 
  - This template helps ensure that each task is clearly defined, actionable, and aligned with the specific goals of your project.

  Template:
  ---------
  def [task_name](self, agent, [parameters]):
      return Task(description=dedent(f'''
      **Task**: [Provide a concise name or summary of the task.]
      **Description**: [Detailed description of what the agent is expected to do, including actionable steps and expected outcomes. This should be clear and direct, outlining the specific actions required to complete the task.]

      **Parameters**: 
      - [Parameter 1]: [Description]
      - [Parameter 2]: [Description]
      ... [Add more parameters as needed.]

      **Note**: [Optional section for incentives or encouragement for high-quality work. This can include tips, additional context, or motivations to encourage agents to deliver their best work.]

      '''), agent=agent)

"""

# This is an example of how to define custom tasks.
# You can define as many tasks as you want.
# You can also define custom agents in agents.py
class TravelTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def plan_itinerary(self, agent, city, travel_dates, interests):
        return Task(
            description=dedent(
                f"""
            **Task**: Desenvolva um itinerário de viagem de 7 dias
            **Description**: [Expanda o guia da cidade em um itinerário de viagem completo de 7 dias com detalhes
            planos diários, incluindo previsões do tempo, lugares para comer, sugestões de embalagem,
            e uma repartição do orçamento. Você DEVE sugerir lugares reais para visitar, hotéis reais para ficar,
            e restaurantes reais para visitar. Este itinerário deve cobrir todos os aspectos da viagem,
            da chegada à saída, integrando as informações do guia da cidade com a prática logística de viagem.
            
            **Parameters**: 
            - Cidade: {city}
            - Data da Viagem: {travel_dates}
            - Interesses dos viajantes: {interests}
            

            **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )

    def identify_city(self, agent, origein, cities, interests, travel_date):
        return Task(
            description=dedent(
                f"""
                    **Task**:  Identify the Best City for the Trip
                    **Description**: Analyze and select the best city for the trip based on specific 
                        criteria such as weather patterns, seasonal events, and travel costs. 
                        This task involves comparing multiple cities, considering factors like current weather 
                        conditions, upcoming cultural or seasonal events, and overall travel expenses. 
                        Your final answer must be a detailed report on the chosen city, 
                        including actual flight costs, weather forecast, and attractions.


                    **Parameters**: 
                    - Origin: {origin}
                    - Cities: {cities}
                    - Interests: {interests}
                    - Travel Date: {travel_dates}

                    **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )
    
    def gather_city_info(self, agent, city, travel_dates, interests):
        return Task(
            description=dedent(
                f"""
                    **Task**:  Gather In-depth City Guide Information
                    **Description**: Compile an in-depth guide for the selected city, gathering information about 
                        key attractions, local customs, special events, and daily activity recommendations. 
                        This guide should provide a thorough overview of what the city has to offer, including 
                        hidden gems, cultural hotspots, must-visit landmarks, weather forecasts, and high-level costs.

                    **Parameters**: 
                    - Cities: {city}
                    - Interests: {interests}
                    - Travel Date: {travel_dates}

                    **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )