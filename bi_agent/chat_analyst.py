"""Conversational BI agent."""
from langchain.agents import create_sql_agent, AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
import plotly.express as px
import pandas as pd

SYSTEM_PROMPT = """You are a senior data analyst. When asked about data:
1. Write precise SQL to answer the question
2. Execute it and interpret results
3. Suggest the best chart type
4. Provide business insights in the same language as the user
Always explain numbers in business context, not just raw data."""

class ConversationalBIAgent:
    def __init__(self, db_uri: str):
        db = SQLDatabase.from_uri(db_uri)
        llm = ChatVertexAI(model_name="gemini-1.5-pro-002")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        self.agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=SYSTEM_PROMPT, verbose=True, max_iterations=10)

    def ask(self, question: str) -> dict:
        result = self.agent.invoke({"input": question})
        return {"answer": result["output"], "question": question}

    def render_chart(self, df: pd.DataFrame, chart_type: str = "bar", x: str = None, y: str = None):
        if chart_type == "bar": return px.bar(df, x=x, y=y)
        if chart_type == "line": return px.line(df, x=x, y=y)
        if chart_type == "scatter": return px.scatter(df, x=x, y=y)
        return px.bar(df)
