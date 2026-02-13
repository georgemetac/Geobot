import argparse
import asyncio
import json
import logging
import math
import os
import random
import statistics
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import dotenv

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# ----------------- math utilities -----------------

def logit(p):
    p = min(1 - 1e-12, max(1e-12, p))
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def extremize(p, factor=1.25):
    return min(0.99, max(0.01, sigmoid(logit(p) * factor)))


# ----------------- aggregation -----------------

def median(xs):
    return statistics.median(xs)


def trimmed_mean(xs, trim=0.2):
    xs = sorted(xs)
    n = len(xs)
    k = int(n * trim)
    if n - 2 * k <= 0:
        return sum(xs) / len(xs)
    return sum(xs[k:-k]) / len(xs[k:-k])


def log_odds_mean(xs):
    return sigmoid(sum(logit(x) for x in xs) / len(xs))


def bayesian_weighted(xs, weights):
    logits = [logit(x) for x in xs]
    weighted = sum(w * l for w, l in zip(weights, logits)) / sum(weights)
    return sigmoid(weighted)


# ----------------- Monte Carlo -----------------

def monte_carlo_binary(base_prob, volatility=0.1, sims=3000):
    samples = []
    for _ in range(sims):
        noise = random.gauss(0, volatility)
        p = sigmoid(logit(base_prob) + noise)
        samples.append(p)
    return sum(samples) / len(samples)


# ----------------- search -----------------

class ExaSearcher:
    def __init__(self):
        self.key = os.getenv("EXA_API_KEY")

    async def search(self, query):
        if not self.key:
            return []
        payload = {"query": query, "num_results": 5}
        req = Request(
            "https://api.exa.ai/search",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-api-key": self.key},
            method="POST",
        )
        resp = await asyncio.to_thread(urlopen, req)
        return json.loads(resp.read().decode()).get("results", [])


class LinkupSearcher:
    def __init__(self):
        self.key = os.getenv("LINKUP_API_KEY")

    async def search(self, query):
        if not self.key:
            return []
        req = Request(
            f"https://api.linkup.so/v1/search?{urlencode({'q': query})}",
            headers={"Authorization": f"Bearer {self.key}"},
        )
        resp = await asyncio.to_thread(urlopen, req)
        return json.loads(resp.read().decode()).get("results", [])


# ----------------- main bot -----------------

class BayesianMonteBot(ForecastBot):

    def __init__(self, aggregation="log_odds", *args, **kwargs):
        llms = {
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5.1",
                temperature=0.25,
                timeout=60,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-5.1",
                temperature=0,
                timeout=60,
                allowed_tries=2,
            ),
        }
        super().__init__(*args, llms=llms, **kwargs)

        self.aggregation = aggregation
        self.exa = ExaSearcher()
        self.linkup = LinkupSearcher()

        self.performance_history = []

    async def run_research(self, question: MetaculusQuestion):
        queries = [
            question.question_text,
            question.question_text + " base rate",
            question.question_text + " prediction market",
        ]
        snippets = []
        for q in queries:
            exa = await self.exa.search(q)
            link = await self.linkup.search(q)
            for r in (exa + link)[:3]:
                snippets.append(f"{r.get('title','')} {r.get('url','')}")
            await asyncio.sleep(1.0)
        return "\n".join(snippets)

    async def _run_forecast_on_binary(self, question, research):
        prompt = f"""
        Question: {question.question_text}
        Research:
        {research}

        Provide reasoning and end with:
        Probability: XX%
        """
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        parsed = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
        )

        return ReasonedPrediction(
            prediction_value=parsed.prediction_in_decimal,
            reasoning=reasoning,
        )

    def aggregate_predictions(self, preds):
        xs = [p.prediction_value for p in preds]

        if self.aggregation == "median":
            base = median(xs)
        elif self.aggregation == "trimmed":
            base = trimmed_mean(xs)
        elif self.aggregation == "bayesian":
            if self.performance_history:
                weights = [
                    math.exp(-score) for score in self.performance_history[-len(xs):]
                ]
            else:
                weights = [1] * len(xs)
            base = bayesian_weighted(xs, weights)
        else:
            base = log_odds_mean(xs)

        base = monte_carlo_binary(base)
        base = extremize(base)
        return base

    async def forecast_questions(self, questions, return_exceptions=False):
        reports = []
        for q in questions:
            research = await self.run_research(q)
            preds = []
            for _ in range(self.predictions_per_research_report):
                pred = await self._run_forecast_on_binary(q, research)
                preds.append(pred)
            final_prob = self.aggregate_predictions(preds)
            final = ReasonedPrediction(
                prediction_value=final_prob,
                reasoning="Aggregated with Bayesian + Monte Carlo",
            )
            reports.append(final)
        return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", choices=["median", "trimmed", "log_odds", "bayesian"], default="log_odds")
    args = parser.parse_args()

    bot = BayesianMonteBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        aggregation=args.aggregation,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    client = MetaculusClient()

    seasonal = asyncio.run(
        bot.forecast_on_tournament(
            client.CURRENT_AI_COMPETITION_ID,
            return_exceptions=True,
        )
    )

    mini = asyncio.run(
        bot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID,
            return_exceptions=True,
        )
    )

    bot.log_report_summary(seasonal + mini)
