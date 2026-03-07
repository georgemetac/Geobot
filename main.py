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
from typing import Any, Dict, List, Optional, Tuple
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


# =============================================================================
# JSON / output sanitisation
# =============================================================================

def sanitize_llm_json(text: str) -> str:
    """
    Clean common LLM JSON formatting issues:
    - remove numeric underscores  (1_000 to 1000)
    - coerce quoted numerics for known fields
    - strip ```json fences
    """
    if not text:
        return ""
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f'"{match.group(1)}": {nums[0]}' if nums else match.group(0)

    text = re.sub(
        r'"(value|percentile|probability|prediction_in_decimal'
        r'|revised_prediction_in_decimal)":\s*"([^"]+)"',
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
    Accept percentiles as 10/20/... or 0.1/0.2/...
    Normalise later to forecasting_tools.Percentile (percentile in [0,1]).
    """
    percentile: float = Field(...)
    value: float


# =============================================================================
# Aggregation helpers
# =============================================================================

def logit(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, float(p)))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def extremize(p: float, factor: float = 1.25) -> float:
    """Push probability away from 0.5 via logit scaling."""
    return min(0.99, max(0.01, sigmoid(logit(p) * factor)))


def agg_median(xs: List[float]) -> float:
    return float(statistics.median(xs))


def agg_trimmed_mean(xs: List[float], trim: float = 0.2) -> float:
    xs = sorted(xs)
    n = len(xs)
    k = int(n * trim)
    if n - 2 * k <= 0:
        return float(sum(xs) / len(xs))
    return float(sum(xs[k: n - k]) / (n - 2 * k))


def agg_log_odds_mean(xs: List[float]) -> float:
    return float(sigmoid(sum(logit(x) for x in xs) / len(xs)))


def agg_bayesian_weighted(xs: List[float], weights: List[float]) -> float:
    logits = [logit(x) for x in xs]
    total_w = max(1e-12, sum(weights))
    weighted = sum(w * l for w, l in zip(weights, logits)) / total_w
    return float(sigmoid(weighted))


def monte_carlo_binary(base_prob: float, volatility: float = 0.1, sims: int = 3000) -> float:
    """Smooth a probability estimate via logit-space Gaussian noise."""
    samples = [
        sigmoid(logit(base_prob) + random.gauss(0, volatility))
        for _ in range(sims)
    ]
    return float(sum(samples) / len(samples))


# =============================================================================
# Reasoning trace  (IMPROVEMENT 2)
# =============================================================================

class ReasoningTrace:
    """
    Accumulates every step of Geob's decision process and renders it as
    a human-readable block embedded in every ReasonedPrediction.
    """

    def __init__(self, question_text: str, bot_name: str = "geob"):
        self.bot_name = bot_name
        self.question_text = question_text
        self._steps: List[Tuple[str, str]] = []

    def add(self, label: str, detail: str) -> None:
        self._steps.append((label, str(detail)))
        logger.info(f"[{self.bot_name}] {label}: {str(detail)[:200]}")

    def add_draft(self, index: int, prob: float, comment: str, narrative: str = "") -> None:
        """Record one draft forecast including the LLM's own reasoning."""
        detail = f"prob={prob:.4f} | comment: {comment}"
        if narrative:
            trimmed = narrative.strip()[:600]
            if len(narrative.strip()) > 600:
                trimmed += "\n... [truncated]"
            detail += f"\n--- LLM reasoning ---\n{trimmed}"
        self._steps.append((f"Draft {index}", detail))
        logger.debug(f"[{self.bot_name}] Draft {index}: prob={prob:.4f}")

    def render(self) -> str:
        lines = [
            f"=== [{self.bot_name.upper()}] REASONING TRACE ===",
            f"  Question : {self.question_text[:120]}",
            f"  Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "--- STEPS ---",
        ]
        for i, (label, detail) in enumerate(self._steps, 1):
            lines.append("")
            lines.append(f"  [{i:02d}] {label}")
            for line in detail.splitlines():
                for chunk in [line[j: j + 108] for j in range(0, max(len(line), 1), 108)]:
                    lines.append(f"       {chunk}")
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# =============================================================================
# Search clients
# =============================================================================

class ExaSearcher:
    """Neural search via EXA_API_KEY."""

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
    """Web search via LINKUP_API_KEY."""

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


# =============================================================================
# Decomposition schema + feature flags
# =============================================================================

class DecompositionOutput(BaseModel):
    subquestions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)


@dataclass
class BotFlags:
    enable_decomposition: bool = True
    enable_asknews: bool = True


# =============================================================================
# Main bot -- Geob  (IMPROVEMENT 1: renamed from GeobDuke)
# =============================================================================

class Geob(ForecastBot):
    """
    Geob -- multi-method aggregating superforecaster bot.

    Pipeline (binary):
      research -> N draft predictions -> aggregate (log_odds / median /
      trimmed / bayesian) -> Monte Carlo smoothing -> extremize -> trace

    See module docstring for full explanation.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(
        self,
        aggregation: str = "log_odds",
        bot_name: str = "geob",
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
            "decomposer": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.2,
                timeout=45,
                allowed_tries=2,
            ),
            "summarizer": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.1,
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

        self.asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")

        self.performance_history: List[float] = []

        # IMPROVEMENT 6: research cache keyed by question URL
        self._research_cache: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
        """
        IMPROVEMENT 9: Properly extracts text from AskNews response
        instead of calling str() on a raw object.
        """
        if not self.flags.enable_asknews:
            return []
        if not self.asknews_client_id or not self.asknews_client_secret:
            return []
        try:
            searcher = AskNewsSearcher(
                client_id=self.asknews_client_id,
                client_secret=self.asknews_client_secret,
            )
            result = await searcher.call_preconfigured_version(
                "asknews/news-summaries", query
            )
            # result may be str, dict, or object -- extract text defensively
            if isinstance(result, str):
                text = result[:700]
            elif isinstance(result, dict):
                text = (
                    result.get("summary")
                    or result.get("text")
                    or result.get("content")
                    or json.dumps(result)[:700]
                )
            else:
                text = (
                    getattr(result, "summary", None)
                    or getattr(result, "text", None)
                    or getattr(result, "content", None)
                    or repr(result)[:700]
                )
            return [{"title": "AskNews", "url": "", "snippet": str(text)[:700]}]
        except Exception as e:
            logger.warning(f"AskNews search failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Research summary  (IMPROVEMENT 5)
    # -------------------------------------------------------------------------

    async def _summarize_research(
        self, question: MetaculusQuestion, snippets_body: str
    ) -> str:
        """
        Generate a concise 3-sentence summary of what the research found.
        Recorded as the first step of every ReasoningTrace.
        """
        llm = self.get_llm("summarizer", "llm")
        prompt = clean_indents(
            f"""
Summarize the web research below for a forecaster. Write exactly 3 sentences:
  1. The most relevant factual finding.
  2. The strongest signal pointing toward YES / a higher value.
  3. The strongest signal pointing toward NO / a lower value.

Be specific -- name figures, dates, sources where present.

Question: {question.question_text}

Research snippets:
{snippets_body[:2500]}
"""
        )
        try:
            return (await llm.invoke(prompt)).strip()
        except Exception as e:
            logger.warning(f"Research summary failed: {e}")
            return "[Research summary unavailable]"

    # -------------------------------------------------------------------------
    # Research  (IMPROVEMENT 6: cached)
    # -------------------------------------------------------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        IMPROVEMENT 6: Results cached by question URL so repeated calls
        (e.g. from multi-draft loops) never re-fetch.
        """
        async with self._concurrency_limiter:
            cache_key = getattr(question, "page_url", None) or question.question_text[:80]
            if cache_key in self._research_cache:
                logger.info(f"[{self.bot_name}] Research cache hit: {cache_key}")
                return self._research_cache[cache_key]

            decomp = await self._decompose(question)

            base_queries = [
                question.question_text,
                question.question_text + " base rate",
                question.question_text + " prediction market",
            ]
            if decomp and decomp.subquestions:
                base_queries.extend(decomp.subquestions[:2])
            if decomp and decomp.key_entities:
                base_queries.append(
                    question.question_text + " " + " ".join(decomp.key_entities[:3])
                )

            queries: List[str] = []
            seen: set = set()
            for q in base_queries:
                q = (q or "").strip()
                if not q or q in seen:
                    continue
                seen.add(q)
                queries.append(q)
            queries = queries[:5]

            snippets: List[str] = []
            used: Dict[str, bool] = {"exa": False, "linkup": False, "asknews": False}

            for q in queries:
                try:
                    exa, link, ask = await asyncio.gather(
                        self.exa.search(q),
                        self.linkup.search(q),
                        self._asknews_search(q),
                        return_exceptions=True,
                    )
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

                    for r in merged[:3]:
                        title = (r.get("title") or "").strip()
                        url = (r.get("url") or "").strip()
                        snippet = (r.get("snippet") or r.get("content") or "").strip()[:200]
                        if title or url:
                            line = f"- {title} {url}".strip()
                            if snippet:
                                line += f"\n  {snippet}"
                            snippets.append(line)
                except Exception:
                    pass
                await asyncio.sleep(0.6)

            footprint = self._search_footprint(used)
            header = f"[research sources: {footprint}]"
            body = "\n".join(snippets[:20]).strip()
            if not body:
                body = "- (no external snippets retrieved; relying on model reasoning)"

            result = header + "\n" + body
            self._research_cache[cache_key] = result
            return result

    # -------------------------------------------------------------------------
    # Numeric helpers
    # -------------------------------------------------------------------------

    def _normalize_raw_percentiles(self, raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p = p / 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        out.sort(key=lambda x: float(x.percentile))
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

    # -------------------------------------------------------------------------
    # Forecasting prompts  (IMPROVEMENT 3: narrative captured per draft)
    # -------------------------------------------------------------------------

    async def _single_binary_draft(
        self,
        question: BinaryQuestion,
        research: str,
        draft_index: int,
        trace: ReasoningTrace,
    ) -> float:
        """
        IMPROVEMENT 3: Returns probability AND captures the LLM's full
        chain-of-thought narrative in the ReasoningTrace.
        """
        prompt = clean_indents(
            f"""
You are a calibrated superforecaster.

Question: {question.question_text}

Resolution criteria:
{question.resolution_criteria}

Research snippets:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Think step by step:
1. What is the base rate for this type of event?
2. What evidence supports YES?
3. What evidence supports NO?
4. What is the status quo if nothing changes?
5. What is your calibrated probability?

Output your reasoning, then end with ONLY this JSON on the final line:
{{"prediction_in_decimal": 0.42, "short_comment": "one sentence on main drivers"}}
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)

        # Separate narrative from final JSON line
        lines = (raw or "").splitlines()
        json_line = ""
        narrative_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{") and "prediction_in_decimal" in stripped:
                json_line = stripped
            else:
                narrative_lines.append(line)
        narrative = "\n".join(narrative_lines).strip()

        prob = 0.5
        comment = ""
        try:
            parsed_dict = json.loads(sanitize_llm_json(json_line or raw))
            prob = float(parsed_dict.get("prediction_in_decimal", 0.5))
            comment = (parsed_dict.get("short_comment") or "").strip()
        except Exception:
            try:
                parsed = await structure_output(
                    sanitize_llm_json(raw),
                    BinaryPrediction,
                    model=self.get_llm("parser", "llm"),
                )
                prob = float(parsed.prediction_in_decimal)
            except Exception as e:
                logger.warning(f"Draft {draft_index} parse failed: {e}")

        prob = float(min(0.99, max(0.01, prob)))
        trace.add_draft(draft_index, prob, comment, narrative)
        return prob

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Single-draft binary forecast (used by ForecastBot base class flow).
        Multi-draft aggregation is handled in forecast_questions override.
        """
        trace = ReasoningTrace(question.question_text, self.bot_name)
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        trace.add("Research sources", research.splitlines()[0] if research else "none")

        prob = await self._single_binary_draft(question, research, 1, trace)
        trace.add("FINAL PREDICTION", f"{prob:.4f}  ({prob:.1%})")
        return ReasonedPrediction(prediction_value=prob, reasoning=trace.render())

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        trace.add("Research sources", research.splitlines()[0] if research else "none")
        trace.add("Options", str(list(question.options)))

        prompt = clean_indents(
            f"""
You are a calibrated superforecaster.

Question: {question.question_text}
Options: {question.options}

Research snippets:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Think step by step:
1. What does the base rate suggest for each option?
2. Which option does the current evidence favour?
3. What is the status quo option?
4. Assign calibrated probabilities summing to exactly 1.0.

Output your reasoning, then end with ONLY this JSON on the final line:
{{
  "predicted_options": [
    {{"option_name": "{question.options[0] if question.options else 'Option A'}", "probability": 0.5}}
  ],
  "short_comment": "one sentence on why the top option leads"
}}
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)

        narrative_lines = [
            ln for ln in (raw or "").splitlines()
            if not ln.strip().startswith("{")
        ]
        trace.add("LLM narrative", "\n".join(narrative_lines).strip()[:800])

        short_comment = ""
        try:
            d = json.loads(sanitize_llm_json(raw))
            short_comment = (d.get("short_comment") or "").strip()
            final_list = PredictedOptionList.model_validate(d)
        except Exception:
            final_list = await structure_output(
                sanitize_llm_json(raw),
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
            )

        trace.add("MC result", short_comment or "parsed from LLM output")
        trace.add(
            "FINAL PREDICTION",
            " | ".join(
                f"{o.option_name}={o.probability:.1%}"
                for o in (final_list.predicted_options or [])
            ),
        )
        return ReasonedPrediction(prediction_value=final_list, reasoning=trace.render())

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        trace = ReasoningTrace(question.question_text, self.bot_name)
        research_summary = await self._summarize_research(question, research)
        trace.add("Research summary", research_summary)
        trace.add("Research sources", research.splitlines()[0] if research else "none")

        units = question.unit_of_measure or "not stated"
        upper = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        lower = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )

        prompt = clean_indents(
            f"""
You are a calibrated superforecaster.

Question: {question.question_text}
Units: {units}
Bounds: [{lower}, {upper}]

Research snippets:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

Think step by step:
1. What is the reference class / historical base rate for this quantity?
2. What trend does the research suggest?
3. What are the key upside risks?
4. What are the key downside risks?
5. How wide should the uncertainty interval be?

Output your reasoning, then end with ONLY these 6 lines and nothing after:
Percentile 10: <number>
Percentile 20: <number>
Percentile 40: <number>
Percentile 60: <number>
Percentile 80: <number>
Percentile 90: <number>
"""
        )
        raw = await self.get_llm("default", "llm").invoke(prompt)

        narrative_lines = []
        for line in (raw or "").splitlines():
            if re.match(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, re.IGNORECASE):
                break
            narrative_lines.append(line)
        trace.add("LLM narrative", "\n".join(narrative_lines).strip()[:800])

        pcts: List[Percentile] = []
        try:
            raw_pcts: List[RawPercentile] = await structure_output(
                sanitize_llm_json(raw),
                list[RawPercentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            pcts = self._normalize_raw_percentiles(raw_pcts)
            pcts = self._require_standard_percentiles(pcts) or pcts
        except Exception as e:
            logger.warning(f"Numeric parse attempt 1 failed: {e}")
            try:
                pcts = await structure_output(
                    sanitize_llm_json(raw),
                    list[Percentile],
                    model=self.get_llm("parser", "llm"),
                )
            except Exception as e2:
                logger.warning(f"Numeric parse attempt 2 failed: {e2}")
                lo_f = float(lower) if lower is not None else 0.0
                hi_f = float(upper) if upper is not None else 1.0
                if not (lo_f < hi_f):
                    lo_f, hi_f = 0.0, 1.0
                w = {0.1: 0.05, 0.2: 0.15, 0.4: 0.40, 0.6: 0.60, 0.8: 0.85, 0.9: 0.95}
                pcts = [
                    Percentile(percentile=p, value=lo_f + (hi_f - lo_f) * w[p])
                    for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                ]
                trace.add("Numeric parse", "FALLBACK -- bounds-based percentiles used")

        distribution = NumericDistribution.from_question(pcts, question)

        try:
            by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
            med = (
                0.5 * (by[0.4] + by[0.6])
                if (0.4 in by and 0.6 in by)
                else float(pcts[len(pcts) // 2].value)
            )
        except Exception:
            med = 0.0

        pct_summary = " | ".join(
            f"P{int(round(float(p.percentile) * 100))}={p.value:.4g}" for p in pcts
        )
        trace.add("Parsed percentiles", pct_summary)
        trace.add("Distribution summary", f"median approx {med:.6g}")
        trace.add("FINAL PREDICTION", pct_summary)
        return ReasonedPrediction(prediction_value=distribution, reasoning=trace.render())

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_forecast_on_numeric(question, research)

    async def _run_forecast_on_conditional(self, question: Any, research: str) -> Any:
        raise NotImplementedError("Conditional questions are not supported by Geob.")

    # -------------------------------------------------------------------------
    # Aggregation  (IMPROVEMENT 4: full trace of aggregation steps)
    # -------------------------------------------------------------------------

    def aggregate_binary(
        self, probs: List[float], trace: ReasoningTrace
    ) -> float:
        """
        IMPROVEMENT 4: Accepts and populates a ReasoningTrace so the
        aggregation method, intermediate value, MC result, and final
        extremized probability are all visible in the output.
        """
        if not probs:
            trace.add("Aggregation", "no drafts -- defaulting to 0.5")
            return 0.5

        trace.add(
            "Draft probabilities",
            f"[{', '.join(f'{p:.4f}' for p in probs)}]",
        )

        if self.aggregation == "median":
            base = agg_median(probs)
        elif self.aggregation == "trimmed":
            base = agg_trimmed_mean(probs)
        elif self.aggregation == "bayesian":
            if self.performance_history:
                tail = self.performance_history[-len(probs):]
                weights = [math.exp(-score) for score in tail]
                while len(weights) < len(probs):
                    weights.insert(0, 1.0)
                weights = weights[-len(probs):]
            else:
                weights = [1.0] * len(probs)
            base = agg_bayesian_weighted(probs, weights)
            trace.add("Bayesian weights", str([f"{w:.3f}" for w in weights]))
        else:
            base = agg_log_odds_mean(probs)

        trace.add(f"Aggregation ({self.aggregation})", f"{base:.4f}")

        mc = monte_carlo_binary(base)
        trace.add(
            "Monte Carlo smoothing (3000 sims, sigma=0.1 logit)",
            f"{base:.4f} -> {mc:.4f}",
        )

        final = extremize(mc)
        trace.add("Extremize (logit x 1.25)", f"{mc:.4f} -> {final:.4f}")
        return float(final)

    # -------------------------------------------------------------------------
    # Main forecast loop  (IMPROVEMENT 7: all types get full traces)
    # -------------------------------------------------------------------------

    async def forecast_questions(
        self,
        questions: List[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        IMPROVEMENT 7: All question types now produce full ReasoningTraces.
        MC and Numeric no longer silently overwrite reasoning with a stub.
        """
        reports: List[Any] = []

        for q in questions:
            try:
                research = await self.run_research(q)

                if isinstance(q, BinaryQuestion):
                    trace = ReasoningTrace(q.question_text, self.bot_name)

                    # IMPROVEMENT 5: research summary as first trace step
                    research_summary = await self._summarize_research(q, research)
                    trace.add("Research summary", research_summary)
                    trace.add(
                        "Research sources",
                        research.splitlines()[0] if research else "none",
                    )
                    trace.add(
                        "Pipeline",
                        f"drafts={self.predictions_per_research_report} | "
                        f"aggregation={self.aggregation} | MC | extremize",
                    )

                    probs: List[float] = []
                    for i in range(self.predictions_per_research_report):
                        p = await self._single_binary_draft(q, research, i + 1, trace)
                        probs.append(p)

                    # IMPROVEMENT 4: aggregation trace
                    final_prob = self.aggregate_binary(probs, trace)
                    trace.add("FINAL PREDICTION", f"{final_prob:.4f}  ({final_prob:.1%})")

                    reports.append(
                        ReasonedPrediction(
                            prediction_value=final_prob,
                            reasoning=trace.render(),
                        )
                    )

                elif isinstance(q, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(q, research)
                    reports.append(pred)

                elif isinstance(q, (NumericQuestion, DateQuestion)):
                    pred = await self._run_forecast_on_numeric(q, research)
                    reports.append(pred)

                else:
                    raise TypeError(f"Unsupported question type: {type(q)}")

            except Exception as e:
                if return_exceptions:
                    reports.append(e)
                else:
                    raise

        return reports


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="geob: Exa+Linkup+AskNews, GPT-5.1, multi-draft aggregation, full reasoning trace"
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "trimmed", "log_odds", "bayesian"],
        default="log_odds",
    )
    parser.add_argument("--bot-name", default="geob")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-asknews", action="store_true")
    args = parser.parse_args()

    bot = Geob(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        aggregation=args.aggregation,
        bot_name=args.bot_name,
        flags=BotFlags(
            enable_decomposition=not args.no_decomposition,
            enable_asknews=not args.no_asknews,
        ),
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
