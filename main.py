import os
import requests
import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal
import dotenv
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

# Load environment variables
dotenv.load_dotenv()

# API Keys from environment
METACULUS_API_TOKEN = os.getenv("METACULUS_API_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")

# Configure logging
logger = logging.getLogger(__name__)

# Metaculus API URL
METACULUS_API_URL = "https://www.metaculus.com/api2"

# OpenRouter API URL
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

class SpringTemplateBot2026:
    """
    Bot that integrates Metaculus and OpenRouter for forecasting and research.
    """

    def __init__(self):
        self.metaculus_token = METACULUS_API_TOKEN
        self.openrouter_key = OPENROUTER_API_KEY
        self.exa_api_key = EXA_API_KEY
        self.linkup_api_key = LINKUP_API_KEY

    ##############################
    # Metaculus API Integration ##
    ##############################

    def get_metaculus_tournament_questions(self, tournament_id: str):
        """
        Fetches questions from a specific Metaculus tournament using their API.
        """
        url = f"{METACULUS_API_URL}/tournaments/{tournament_id}/questions/"
        headers = {
            "Authorization": f"Bearer {self.metaculus_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Check if request was successful
            questions = response.json()
            logger.info(f"Fetched {len(questions)} questions from tournament {tournament_id}.")
            return questions
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Metaculus API: {e}")
            return []

    ##############################
    # OpenRouter API Integration ##
    ##############################

    def query_openrouter(self, model: str, prompt: str) -> str:
        """
        Sends a query to OpenRouter for a given model (GPT-5.1 or Claude-Sonnet-4.5).
        """
        url = f"{OPENROUTER_API_URL}/models/{model}/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 150
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Check if request was successful
            completion = response.json()
            return completion.get("choices", [{}])[0].get("text", "No completion returned.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying OpenRouter API: {e}")
            return "Error: Unable to fetch data from OpenRouter."

    ##############################
    # Exa API Integration ##
    ##############################

    def query_exa_api(self, instructions: str) -> str:
        """
        Query Exa API for deep research.
        """
        url = "https://api.exa.ai/research/v1"
        headers = {
            "x-api-key": self.exa_api_key,
            "Content-Type": "application/json"
        }
        data = {"instructions": instructions, "model": "exa-research"}
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Check if request was successful
            research = response.json()
            return research.get("research", "No research found")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Exa API: {e}")
            return "Error: Unable to fetch data from Exa."

    ##############################
    # LinkUp API Integration ##
    ##############################

    def query_linkup_search(self, query: str) -> str:
        """
        Query LinkUp API for search results.
        """
        url = "https://api.linkup.so/v1/search"
        headers = {
            "Authorization": f"Bearer {self.linkup_api_key}",
            "Content-Type": "application/json"
        }
        data = {"q": query, "depth": "deep", "outputType": "searchResults"}
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Check if request was successful
            search_results = response.json()
            return search_results.get("results", "No results found")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying LinkUp API: {e}")
            return "Error: Unable to fetch data from LinkUp."

    ##############################
    # Research Aggregation ##
    ##############################

    async def perform_research_with_exa_and_linkup(self, prompt: str) -> str:
        """
        Perform research using both Exa and LinkUp, returns combined results.
        """
        exa_research = await asyncio.to_thread(self.query_exa_api, prompt)
        linkup_research = await asyncio.to_thread(self.query_linkup_search, prompt)

        # Combine both results
        combined_research = f"Exa Research:\n{exa_research}\n\nLinkUp Research:\n{linkup_research}"
        return combined_research

    ##############################
    # Forecasting Methods ##
    ##############################

    async def _run_forecast_on_binary(self, question: dict, research: str) -> str:
        base_rate = 0.5  # Example base rate
        prompt = f"""
        You are a professional forecaster.
        Forecast this binary question based on the given research and historical base rates.

        Question:
        {question["question_text"]}

        Research:
        {research}

        Forecasting Principles:
        (a) Start with the base rate (historically, this event has occurred {base_rate*100}% of the time).
        (b) Consider unexpected scenarios and the status quo.
        (c) Provide your probability estimate with reasoning.

        Final Answer: Probability: ZZ%
        Explanation: Based on historical data and the current status, the likelihood of this event occurring is influenced by...
        """
        return await self.query_openrouter("gpt-5.1", prompt)

    async def _run_forecast_on_multiple_choice(self, question: dict, research: str) -> str:
        base_rate = [0.25] * len(question["options"])  # Base rate for each option
        prompt = f"""
        You are a professional forecaster.
        Forecast this multiple choice question based on the given research and historical base rates.

        Question:
        {question["question_text"]}

        Research:
        {research}

        Forecasting Principles:
        (a) Start with the base rate for each option.
        (b) Incorporate the status quo for each option.
        (c) Adjust for unexpected scenarios for each option.

        Final Answer:
        Option_A: Probability_A
        Option_B: Probability_B
        ...
        Option_N: Probability_N
        Explanation: The probabilities for each option are calculated based on...
        """
        return await self.query_openrouter("gpt-5.1", prompt)

    ##############################
    # Example Method for Bot ##
    ##############################

    async def run_research_and_forecast(self, question: dict):
        """
        Combines research and forecasting for a given question.
        """
        # Fetch research from Exa and LinkUp
        research = await self.perform_research_with_exa_and_linkup(question["question_text"])

        # Perform forecasting for different question types (binary, multiple choice, etc.)
        if question["type"] == "binary":
            forecast = await self._run_forecast_on_binary(question, research)
        elif question["type"] == "multiple_choice":
            forecast = await self._run_forecast_on_multiple_choice(question, research)

        logger.info(f"Forecast for {question['question_text']}: {forecast}")
        return forecast


# Example of bot instantiation and running research for specific tournaments
if __name__ == "__main__":
    bot = SpringTemplateBot2026()

    # Tournament IDs: 32916 and minibench
    tournament_ids = [32916, "minibench"]

    # Fetch questions from specific tournaments and run forecasts
    for tournament_id in tournament_ids:
        questions = bot.get_metaculus_tournament_questions(tournament_id)

        for question in questions:
            example_question = {
                "question_text": question.get("question_text"),
                "type": "binary",  # Assuming binary questions for simplicity
                "options": ["Yes", "No"]
            }
            # Run research and forecasting asynchronously
            asyncio.run(bot.run_research_and_forecast(example_question))

