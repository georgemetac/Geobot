import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import dotenv
import httpx
from pydantic import BaseModel, Field

from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# =========================================================
# 🛡️ SANITIZATION / SAFE PARSING
# =========================================================

def sanitize_llm_json(text: str) -> str:
    """
    Clean common LLM JSON formatting issues:
    - remove numeric underscores (1_000 -> 1000)
    - coerce quoted numerics for a few known fields
    - remove ```json fences
    """
    if not text:
        return ""
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f"\"{match.group(1)}\": {nums[0]}" if nums else match.group(0)

    text = re.sub(
        r"\"(value|percentile|probability|prediction_in_decimal|revised_prediction_in_decimal)\":\s*\"([^\"]+)\"",
        clean_num,
        text,
    )

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class RawPercentile(BaseModel):
    """
    Accept percentiles as 10/20/... or 0.1/0.2/... and values as floats.
    Normalize later to forecasting_tools.Percentile (percentile in [0,1]).
    """
    percentile: float = Field(...)
    value: float


# =========================================================
# 🎲 AGGREGATION HELPERS (maintains your original logic)
# =========================================================

def logit(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, float(p)))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def extremize(p: float, factor: float = 1.25) -> float:
    return min(0.99, max(0.01, sigmoid(logit(p) * factor)))


def median(xs: List[float]) -> float:
    return float(statistics.median(xs))


def trimmed_mean(xs: List[float], trim: float = 0.2) -> float:
    xs = sorted(xs)
    n = len(xs)
    k = int(n * trim)
    if n - 2 * k <= 0:
        return float(sum(xs) / len(xs))
    return float(sum(xs[k:-k]) / len(xs[k:-k]))


def log_odds_mean(xs: List[float]) -> float:
    return float(sigmoid(sum(logit(x) for x in xs) / len(xs)))


def bayesian_weighted(xs: List[float], weights: List[float]) -> float:
    logits = [logit(x) for x in xs]
    weighted = sum(w * l for w, l in zip(weights, logits)) / max(1e-12, sum(weights))
    return float(sigmoid(weighted))


def monte_carlo_binary(base_prob: float, volatility: float = 0.1, sims: int = 3000) -> float:
    samples = []
    for _ in range(sims):
        noise = random.gauss(0, volatility)
        p = sigmoid(logit(base_prob) + noise)
        samples.append(p)
    return float(sum(samples) / len(samples))


# =========================================================
# 🔎 SEARCH CLIENTS (same providers, more robust)
# =========================================================

class ExaSearcher:
    """
    Keeps your Exa logic but uses httpx async and handles errors cleanly.
    """
    def __init__(self):
        self.key = os.getenv("EXA_API_KEY")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 5) -> List[dict]:
        if not self.key:
            return []
        payload = {"query": query, "numResults": num_results}
        headers = {"Content-Type": "application/json", "x-api-key": self.key}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(self.base_url, json=payload, headers=headers)
                r.raise_for_status()
                return r.json().get("results", []) or []
        except Exception as e:
            logger.warning(f"Exa search failed: {e}")
            return []


class LinkupSearcher:
    """
    Keeps your Linkup logic but uses httpx async and handles errors cleanly.
    """
    def __init__(self):
        self.key = os.getenv("LINKUP_API_KEY")
        self.base_url = "https://api.linkup.so/v1/search"

    async def search(self, query: str) -> List[dict]:
        if not self.key:
            return []
        try:
            qs = urlencode({"q": query})
            headers = {"Authorization": f"Bearer {self.key}"}
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(f"{self.base_url}?{qs}", headers=headers)
                r.raise_for_status()
                return r.json().get("results", []) or []
        except Exception as e:
            logger.warning(f"Linkup search failed: {e}")
            return []


# =========================================================
# 🧩 OPTIONAL: LIGHT DECOMPOSITION (only to improve research)
# =========================================================

class DecompositionOutput(BaseModel):
    subquestions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)


@dataclass
class BotFlags:
    enable_decomposition: bool = True
    enable_asknews: bool = True  # optional extra provider (doesn't change your aggregation logic)


# =========================================================
# 🤖 GEO(B) DUKE VERSION (same logic; improved structure/output)
# =========================================================

class GeobDuke(ForecastBot):
    """
    Maintains your original forecasting logic:
      - multiple raw predictions per question
      - aggregate via (median / trimmed / log_odds / bayesian)
      - then monte-carlo smoothing
      - then extremize

    Improvements added (format + reliability):
      - robust async search clients
      - optional lightweight decomposition to improve research queries
      - optional AskNews integration (if credentials exist)
      - safe JSON sanitation for model outputs
      - short reasoning string that includes approach + final forecast for Metaculus
      - keeps concurrency limiter
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(
        self,
        aggregation: str = "log_odds",
        bot_name: str = "botduke",
        flags: Optional[BotFlags] = None,
        *args,
        **kwargs,
    ):
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
            # small helper for decomposition/query rewrite
            "decomposer": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.2,
                timeout=45,
                allowed_tries=2,
            ),
        }
        super().__init__(*args, llms=llms, **kwargs)

        self.aggregation = aggregation
        self.bot_name = bot_name
        self.flags = flags or BotFlags()

        self.exa = ExaSearcher()
        self.linkup = LinkupSearcher()

        # optional AskNews
        self.asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")

        # same idea you had; used for bayesian weighting if you update it
        self.performance_history: List[float] = []

    # -------------------------
    # Small helpers
    # -------------------------

    def _search_footprint(self, used: Dict[str, bool]) -> str:
        srcs = [k for k, v in used.items() if v]
        return ",".join(srcs) if srcs else "none"

    async def _decompose(self, q: MetaculusQuestion) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition:
            return None
        try:
            llm = self.get_llm("decomposer", "llm")
            prompt = clean_indents(
                f"""
Decompose this forecasting question to help search & research.
Return ONLY JSON:
{{"subquestions":[...], "key_entities":[...], "key_metrics":[...]}}

Question:
{q.question_text}

Resolution criteria:
{q.resolution_criteria}
"""
            )
            raw = await llm.invoke(prompt)
            return DecompositionOutput.model_validate_json(sanitize_llm_json(raw))
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            return None

    async def _asknews_search(self, query: str) -> List[dict]:
        if not self.flags.enable_asknews:
            return []
        if not self.asknews_client_id or not self.asknews_client_secret:
            return []
        try:
            searcher = AskNewsSearcher(client_id=self.asknews_client_id, client_secret=self.asknews_client_secret)
            result = await searcher.call_preconfigured_version("asknews/news-summaries", query)
            # normalize into list-of-dicts-ish
            return [{"title": "AskNews", "url": "", "snippet": str(result)[:700]}]
        except Exception as e:
            logger.warning(f"AskNews search failed: {e}")
            return []

    # -------------------------
    # Research (keeps your original intent: snippets list)
    # -------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            decomp = await self._decompose(question)

            base_queries = [
                question.question_text,
                question.question_text + " base rate",
                question.question_text + " prediction market",
            ]

            # add a couple decomposition-driven queries (doesn't change logic; just helps search)
            if decomp and decomp.subquestions:
                base_queries.extend(decomp.subquestions[:2])
            if decomp and decomp.key_entities:
                base_queries.append(question.question_text + " " + " ".join(decomp.key_entities[:3]))

            # de-dup, keep short
            queries = []
            seen = set()
            for q in base_queries:
                q = (q or "").strip()
                if not q or q in seen:
                    continue
                seen.add(q)
                queries.append(q)
            queries = queries[:5]

            snippets: List[str] = []
            used = {"exa": False, "linkup": False, "asknews": False}

            for q in queries:
                try:
                    exa_task = self.exa.search(q)
                    link_task = self.linkup.search(q)
                    ask_task = self._asknews_search(q)

                    exa, link, ask = await asyncio.gather(exa_task, link_task, ask_task, return_exceptions=True)

                    if isinstance(exa, list) and exa:
                        used["exa"] = True
                    if isinstance(link, list) and link:
                        used["linkup"] = True
                    if isinstance(ask, list) and ask:
                        used["asknews"] = True

                    merged: List[dict] = []
                    for block in (exa, link, ask):
                        if isinstance(block, list):
                            merged.extend(block)

                    # keep your original style: title + url only (short)
                    for r in merged[:3]:
                        title = (r.get("title") or "").strip()
                        url = (r.get("url") or "").strip()
                        if title or url:
                            snippets.append(f"- {title} {url}".strip())
                except Exception:
                    pass

                # keep your pacing; but slightly shorter
                await asyncio.sleep(0.6)

            header = f"[research sources: {self._search_footprint(used)}]"
            body = "\n".join(snippets[:20]).strip()
            if not body:
                body = "- (no external snippets retrieved; relying on model reasoning)"
            return header + "\n" + body

    # -------------------------
    # Forecasting prompts (same logic; better structured output)
    # -------------------------

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # Keep your original behaviour: ask for reasoning and probability.
        # But now enforce JSON output for reliability.
        prompt = clean_indents(
            f"""
You are a careful forecaster.

Question: {question.question_text}

Research snippets (may be sparse):
{research}

Return ONLY JSON:
{{
  "prediction_in_decimal": 0.42,
  "short_comment": "1 sentence on main drivers / uncertainty"
}}
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)
        raw = sanitize_llm_json(raw)

        # Parse probability robustly via BinaryPrediction
        # If the model includes extra keys, we parse dict then extract.
        try:
            parsed_dict = json.loads(raw)
            prob = float(parsed_dict.get("prediction_in_decimal"))
            short_comment = (parsed_dict.get("short_comment") or "").strip()
        except Exception:
            # fallback to structure_output for BinaryPrediction
            parsed = await structure_output(raw, BinaryPrediction, model=self.get_llm("parser", "llm"))
            prob = float(parsed.prediction_in_decimal)
            short_comment = ""

        prob = float(min(0.99, max(0.01, prob)))
        reasoning = f"[{self.bot_name}] single-model draft={prob:.3f}. {short_comment}".strip()
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
You are a careful forecaster.

Question: {question.question_text}
Options: {question.options}

Research snippets:
{research}

Return ONLY JSON with probabilities that sum to 1:
{{
  "predicted_options": [
    {{"option_name": "{question.options[0] if question.options else "Option A"}", "probability": 0.5}}
  ],
  "short_comment": "1 sentence on why the top option leads"
}}
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)
        raw = sanitize_llm_json(raw)

        # Try to parse as dict then feed PredictedOptionList
        short_comment = ""
        try:
            d = json.loads(raw)
            short_comment = (d.get("short_comment") or "").strip()
            final_list = PredictedOptionList.model_validate(d)
        except Exception:
            final_list = await structure_output(raw, PredictedOptionList, model=self.get_llm("parser", "llm"))

        reasoning = f"[{self.bot_name}] MC draft; normalized; {short_comment}".strip()
        return ReasonedPrediction(prediction_value=final_list, reasoning=reasoning)

    def _normalize_raw_percentiles(self, raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p = p / 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        out.sort(key=lambda x: float(x.percentile))
        # enforce monotone
        for i in range(1, len(out)):
            if out[i].value <= out[i - 1].value:
                out[i].value = out[i - 1].value + 1e-6
        return out

    def _require_standard_percentiles(self, pcts: List[Percentile]) -> List[Percentile]:
        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        by = {round(float(p.percentile), 3): p for p in pcts}
        if any(round(r, 3) not in by for r in required):
            return []
        return [by[round(r, 3)] for r in required]

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        # Keep your percentile style, but parse robustly using RawPercentile.
        units = question.unit_of_measure or ""
        prompt = clean_indents(
            f"""
You are a careful forecaster.

Question: {question.question_text}
Units: {units}

Research snippets:
{research}

Return ONLY JSON as a list of 6 objects:
[
  {{"percentile": 10, "value": 1.0}},
  {{"percentile": 20, "value": 2.0}},
  {{"percentile": 40, "value": 4.0}},
  {{"percentile": 60, "value": 6.0}},
  {{"percentile": 80, "value": 8.0}},
  {{"percentile": 90, "value": 9.0}}
]
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)
        raw = sanitize_llm_json(raw)

        # Parse list[RawPercentile] with structure_output (tolerant)
        try:
            raw_pcts: List[RawPercentile] = await structure_output(
                raw,
                list[RawPercentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            pcts = self._normalize_raw_percentiles(raw_pcts)
            pcts = self._require_standard_percentiles(pcts) or pcts
        except Exception as e:
            logger.warning(f"Numeric parse failed, falling back to forecasting_tools.Percentile parsing: {e}")
            pcts = await structure_output(raw, list[Percentile], model=self.get_llm("parser", "llm"))

        distribution = NumericDistribution.from_question(pcts, question)

        # a short “median-ish” proxy to report
        try:
            by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
            med = 0.5 * (by.get(0.4, 0.0) + by.get(0.6, 0.0)) if (0.4 in by and 0.6 in by) else float(pcts[len(pcts)//2].value)
        except Exception:
            med = 0.0

        reasoning = f"[{self.bot_name}] numeric draft; parsed percentiles; median≈{med:g}"
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)

    async def _run_forecast_on_date(self, question: DateQuestion, research: str):
        # preserve your original mapping
        return await self._run_forecast_on_numeric(question, research)

    async def _run_forecast_on_conditional(self, question, research):
        raise NotImplementedError

    # -------------------------
    # Aggregation (unchanged logic; better short reasoning)
    # -------------------------

    def aggregate_binary(self, preds: List[ReasonedPrediction[float]]) -> float:
        xs = [float(p.prediction_value) for p in preds]
        if not xs:
            return 0.5

        if self.aggregation == "median":
            base = median(xs)
        elif self.aggregation == "trimmed":
            base = trimmed_mean(xs)
        elif self.aggregation == "bayesian":
            if self.performance_history:
                # Keep your original intent, but guard length mismatch
                tail = self.performance_history[-len(xs):]
                weights = [math.exp(-score) for score in tail] if tail else [1.0] * len(xs)
            else:
                weights = [1.0] * len(xs)
            base = bayesian_weighted(xs, weights)
        else:
            base = log_odds_mean(xs)

        base = monte_carlo_binary(base)
        base = extremize(base)
        return float(base)

    async def forecast_questions(self, questions: List[MetaculusQuestion], return_exceptions: bool = False):
        """
        Maintains your core logic:
          - research once per question
          - generate N draft predictions (binary only) per question
          - aggregate via your selected method + monte carlo + extremize
          - return final ReasonedPrediction with short reasoning + final value

        NOTE: This preserves your behavior even though ForecastBot may have its own
        aggregation flow. If your existing GitHub runner depends on this override,
        keep it as-is.
        """
        reports = []
        for q in questions:
            try:
                research = await self.run_research(q)

                if isinstance(q, BinaryQuestion):
                    preds: List[ReasonedPrediction[float]] = []
                    for _ in range(self.predictions_per_research_report):
                        pred = await self._run_forecast_on_binary(q, research)
                        preds.append(pred)

                    final_prob = self.aggregate_binary(preds)

                    # short comment includes approach + final
                    reasoning = (
                        f"[{self.bot_name}] approach: {self.aggregation}+monte_carlo+extremize; "
                        f"drafts={len(preds)}; final={final_prob:.3f}"
                    )
                    reports.append(ReasonedPrediction(prediction_value=final_prob, reasoning=reasoning))

                elif isinstance(q, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(q, research)
                    # keep short reasoning + rely on parsed structure
                    pred.reasoning = f"[{self.bot_name}] approach: single-pass MC; final distribution returned."
                    reports.append(pred)

                elif isinstance(q, (NumericQuestion, DateQuestion)):
                    pred = await self._run_forecast_on_numeric(q, research)
                    pred.reasoning = f"[{self.bot_name}] approach: single-pass numeric; final distribution returned."
                    reports.append(pred)

                else:
                    raise TypeError(f"Unsupported question type: {type(q)}")

            except Exception as e:
                if return_exceptions:
                    reports.append(e)
                else:
                    raise
        return reports


# =========================================================
# 🚀 MAIN
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregation",
        choices=["median", "trimmed", "log_odds", "bayesian"],
        default="log_odds",
    )
    parser.add_argument("--bot-name", default="botduke")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-asknews", action="store_true")
    args = parser.parse_args()

    bot = GeobDuke(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        aggregation=args.aggregation,
        bot_name=args.bot_name,
        flags=BotFlags(enable_decomposition=not args.no_decomposition, enable_asknews=not args.no_asknews),
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
