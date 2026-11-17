"""Token accounting and explicit rate-card estimates; native billing remains separate."""

from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import re


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens", "total_tokens")
# Counting-semantics strings recorded by different backends (Codex keeps its historical literal; changing it would
# trip resume_model_budget_semantics_changed) map to ONE comparison basis for reports.
COUNTING_EQUIVALENCE = {
    "delivered_usable_actor_helper_compact_responses_v2": "all_delivered_usable_generation_completions",
    "delivered_usable_generation_completions_v1": "all_delivered_usable_generation_completions",
}
RATE_FIELDS = ("uncached_input_per_million", "cached_input_per_million", "output_per_million")


def counter(value):
    return value if type(value) is int and value >= 0 else None


def counting_basis(counting):
    """Comparison basis of a recorded counting-semantics string (``COUNTING_EQUIVALENCE``); an unknown string is its
    own basis and a missing one is None, so strings that are not declared equivalent are never merged."""
    if not isinstance(counting, str) or not counting:
        return None
    return COUNTING_EQUIVALENCE.get(counting, counting)


class RelayUsageTotal:
    """Sum of the normalised usage of every delivered completion seen by the relay (design 5.5), for any backend.

    Each ``add`` takes one completion's counters (``TOKEN_FIELDS`` keys; ``None`` = the call did not report it). A
    total counter becomes None, and stays None, as soon as one call lacks it: unknown is never zero. ``seed`` is a
    previously stored ``runtime/model_usage_relay`` record of the SAME thread (resume); a seed of another thread is
    ignored (``thread_id`` None accepts any seed).
    """

    def __init__(self, seed=None, thread_id=None):
        if isinstance(seed, dict) and isinstance(seed.get("total"), dict) and thread_id in (None, seed.get("thread_id")):
            self.total = {key: seed["total"].get(key) for key in TOKEN_FIELDS}
            self.unknown_calls, self.calls = counter(seed.get("unknown_calls")) or 0, counter(seed.get("calls")) or 0
        else:
            self.total, self.unknown_calls, self.calls = {key: 0 for key in TOKEN_FIELDS}, 0, 0

    def add(self, usage, *, model_context_window=None, thread_id=None, turn_id=None):
        """Accumulate one completion; returns the ``runtime/token_usage``-shaped record
        ``{total, last, model_context_window, thread_id, turn_id, raw_ref: None, unknown_calls, calls}``."""
        usage = usage if isinstance(usage, dict) else {}
        last = {key: counter(usage.get(key)) for key in TOKEN_FIELDS}
        self.calls += 1
        self.unknown_calls += any(value is None for value in last.values())
        for key in TOKEN_FIELDS:
            self.total[key] = None if self.total[key] is None or last[key] is None else self.total[key] + last[key]
        return dict(total=dict(self.total), last=last, model_context_window=counter(model_context_window) or None,
                    thread_id=thread_id, turn_id=turn_id, raw_ref=None, unknown_calls=self.unknown_calls, calls=self.calls)


def responses_counters(usage):
    """The ``TOKEN_FIELDS`` counters of one Responses ``usage`` object through ``account_responses_usage``' field
    mapping (the Codex measurement the relay shares with every backend); absent counters stay None."""
    tokens = account_responses_usage(usage)["tokens"]
    return {key: tokens[key] for key in TOKEN_FIELDS}


def add_served_model(served, model):
    """``served`` (``{model: completed_calls}``) with one more delivered completion served by ``model``; a response
    that did not name its model is counted under ``"unknown"``. Returns a new dict."""
    key = model if isinstance(model, str) and model else "unknown"
    result = dict(served or {})
    result[key] = (counter(result.get(key)) or 0) + 1
    return result


def load_price_card(path):
    """Freeze a user-supplied price schedule, never guess a model's price."""
    raw = Path(path).expanduser().read_bytes()
    card = json.loads(raw)
    if not isinstance(card, dict): raise ValueError("price_card_must_be_object")
    for key in ("version", "source", "currency"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            raise ValueError(f"price_card_requires_{key}")
    if not re.fullmatch(r"[A-Z]{3}", card["currency"]):
        raise ValueError("price_card_currency_requires_uppercase_ISO_code")
    if "billing_basis" in card and (not isinstance(card["billing_basis"], str) or not card["billing_basis"].strip()):
        raise ValueError("price_card_billing_basis_must_be_nonempty_text")
    if not isinstance(card.get("rates"), list) or not card["rates"]:
        raise ValueError("price_card_requires_rates")
    seen = set()
    for rate in card["rates"]:
        if not isinstance(rate, dict): raise ValueError("price_card_rate_must_be_object")
        for key in ("provider", "model"):
            if not isinstance(rate.get(key), str) or not rate[key].strip():
                raise ValueError(f"price_card_rate_requires_{key}")
        identity = (rate["provider"], rate["model"])
        if identity in seen: raise ValueError("price_card_duplicate_provider_model")
        seen.add(identity)
        for key in RATE_FIELDS:
            value = rate.get(key)
            try:
                number = Decimal(value) if isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else Decimal("NaN")
            except InvalidOperation:
                number = Decimal("NaN")
            if not number.is_finite() or number < 0:
                raise ValueError(f"price_card_requires_nonnegative_decimal_string:{key}")
        # Only ordinary input/cache-read/output pricing is implemented. A provider
        # charging cache writes separately needs an explicit billing adapter.
        if "max_input_tokens" in rate and (type(rate["max_input_tokens"]) is not int or rate["max_input_tokens"] <= 0):
            raise ValueError("price_card_max_input_tokens_must_be_positive_integer")
        if set(rate) - {"provider", "model", *RATE_FIELDS, "max_input_tokens"}:
            raise ValueError("price_card_unknown_rate_field")
    return {**card, "sha256": hashlib.sha256(raw).hexdigest()}


def account_usage(usage, *, model=None, provider=None, price_card=None, source="codex.thread/tokenUsage/updated.total"):
    """Use ONE native cumulative snapshot; reasoning is a subset of output.

    source: provenance label stored in the result (default: the Codex native snapshot literal;
    relay-accumulated backends pass e.g. "relay.<wire>.usage.sum").
    """
    usage = usage if isinstance(usage, dict) else {}
    total = usage.get("total") if isinstance(usage.get("total"), dict) else {}
    tokens = {key: counter(total.get(key)) for key in TOKEN_FIELDS}
    context_window = counter(usage.get("model_context_window")) or None
    incoming, cached = tokens["input_tokens"], tokens["cached_input_tokens"]
    issues = []
    if incoming is not None and cached is not None and cached > incoming:
        issues.append("cached_input_exceeds_input")
    output, reasoning = tokens["output_tokens"], tokens["reasoning_output_tokens"]
    if output is not None and reasoning is not None and reasoning > output:
        issues.append("reasoning_output_exceeds_output")
    native_total = tokens["total_tokens"]
    if incoming is not None and output is not None and native_total is not None and native_total != incoming + output:
        issues.append("total_tokens_mismatch")
    uncached = incoming - cached if incoming is not None and cached is not None and cached <= incoming else None
    writes = tokens["cache_write_input_tokens"]
    if uncached is not None and writes is not None and writes > uncached:
        issues.append("cache_write_exceeds_uncached_input")
    tokens["uncached_input_tokens"] = uncached
    with localcontext() as context:
        context.prec = 50
        hit_rate = str(Decimal(cached) / incoming) if incoming and uncached is not None else None
    estimate = dict(status="unavailable", reason="price_card_not_configured", currency=None,
                    uncached_input=None, cached_input=None, cache_write_input=None, output=None,
                    known_subtotal="0", total=None, price_card=None)
    if price_card is not None:
        estimate.update(currency=price_card["currency"], reason="provider_or_model_unknown" if not model or not provider else "provider_model_price_not_found",
                        price_card={key: price_card[key] for key in ("version", "source", "currency", "sha256")})
        if "billing_basis" in price_card:
            estimate["price_card"]["billing_basis"] = price_card["billing_basis"]
        rate = next((item for item in price_card["rates"] if item["model"] == model and item["provider"] == provider), None)
        if rate:
            estimate["price_card"]["rate"] = rate
            writes = tokens["cache_write_input_tokens"]
            # Missing cache-write counters remain unknown; ordinary components
            # can still be reported separately without claiming a complete bill.
            reason = "invalid_usage_counters" if issues else None
            # Pricing thresholds apply to EACH request, not cumulative thread
            # input. The native context capacity is a conservative upper bound.
            limit = rate.get("max_input_tokens")
            if reason is None and limit is not None:
                bound = counter(usage.get("request_input_tokens")) if usage.get("scope") == "single_request" else context_window
                if bound is None:
                    reason = "request_input_bound_unknown"
                elif bound > limit:
                    reason = "request_input_bound_exceeds_flat_rate_limit"
            if reason:
                estimate["reason"] = reason
            else:
                with localcontext() as context:
                    context.prec = 50
                    ordinary_input = uncached - writes if uncached is not None and writes is not None else None
                    for name, amount, rate_name in (("uncached_input", ordinary_input, "uncached_input_per_million"),
                            ("cached_input", cached, "cached_input_per_million"), ("output", output, "output_per_million")):
                        if amount is not None:
                            estimate[name] = format(Decimal(amount) * Decimal(rate[rate_name]) / 1_000_000, "f")
                    parts = [estimate[name] for name in ("uncached_input", "cached_input", "output")]
                    estimate["known_subtotal"] = format(sum((Decimal(part) for part in parts if part is not None), Decimal(0)), "f")
                    if writes == 0:
                        estimate["cache_write_input"] = "0"
                    if writes != 0:
                        estimate.update(status="partial", reason="cache_write_pricing_unknown")
                    elif all(part is not None for part in parts):
                        estimate.update(status="complete", reason=None, total=estimate["known_subtotal"])
                    else:
                        estimate.update(status="partial", reason="token_counter_missing")
    return dict(schema_version="1", scope="thread_cumulative", source=source,
                thread_id=usage.get("thread_id"), turn_id=usage.get("turn_id"), raw_ref=usage.get("raw_ref"),
                model=model, provider=provider, model_context_window=context_window, tokens=tokens,
                cache_hit_rate=hit_rate, issues=issues, rate_card_estimate=estimate,
                provider_reported=dict(status="unavailable", amount=None, currency=None,
                                       reason="native_token_usage_has_no_billing_amount"))


def aggregate_accounting(rows):
    """Keep coverage and known subtotals so missing/failed runs cannot look free."""
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids): raise ValueError("duplicate_run_id_in_cost_aggregate")
    accounts = [row["token_accounting"] for row in rows]
    tokens = {}
    for key in (*TOKEN_FIELDS, "uncached_input_tokens"):
        values = [item["tokens"][key] for item in accounts]
        known = [value for value in values if value is not None]
        tokens[key] = dict(total=sum(known) if len(known) == len(values) else None,
                           known_subtotal=sum(known), known_runs=len(known), unknown_runs=len(values) - len(known))
    incoming, cached = tokens["input_tokens"]["total"], tokens["cached_input_tokens"]["total"]
    with localcontext() as context:
        context.prec = 50
        hit_rate = str(Decimal(cached) / incoming) if incoming and cached is not None and cached <= incoming else None
        costs = {}
        for item in accounts:
            estimate = item["rate_card_estimate"]
            currency = estimate["currency"] or "unknown"
            group = costs.setdefault(currency, dict(known_subtotal="0", known_runs=0, unknown_runs=0))
            if estimate["total"] is None:
                group["unknown_runs"] += 1
            else:
                group["known_runs"] += 1
                group["known_subtotal"] = format(Decimal(group["known_subtotal"]) + Decimal(estimate["total"]), "f")
        for group in costs.values():
            group["total"] = group["known_subtotal"] if not group["unknown_runs"] else None
    return dict(run_count=len(rows), tokens=tokens, cache_hit_rate=hit_rate,
                rate_card_estimate_by_currency=costs,
                provider_reported=dict(amount=None, reason="native_token_usage_has_no_billing_amount"))


def account_responses_usage(usage, *, model=None, provider=None, price_card=None):
    """Adapt one Responses attempt; absent counters and failed usage stay unknown."""
    usage = usage if isinstance(usage, dict) else {}
    incoming = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    outgoing = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    total = dict(input_tokens=usage.get("input_tokens"), cached_input_tokens=incoming.get("cached_tokens"),
                 cache_write_input_tokens=incoming.get("cache_write_tokens", usage.get("cache_write_input_tokens")),
                 output_tokens=usage.get("output_tokens"), reasoning_output_tokens=outgoing.get("reasoning_tokens"),
                 total_tokens=usage.get("total_tokens"))
    snapshot = dict(total=total, scope="single_request", request_input_tokens=usage.get("input_tokens"))
    result = account_usage(snapshot, model=model, provider=provider, price_card=price_card)
    result.update(scope="single_request", source="Responses.response.usage",
                  request_input_tokens=counter(usage.get("input_tokens")))
    return result


def mixed_accounting(entries):
    """Sum distinct accounting units, grouped by model/source; never add snapshots twice."""
    identifiers = [entry["accounting_id"] for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate_mixed_accounting_id")
    groups = {}
    for entry in entries:
        account = entry["accounting"]
        key = (account["model"], account["provider"], entry["source"])
        groups.setdefault(key, []).append(entry)
    def summarize(values):
        # Existing aggregation preserves token coverage and complete-cost coverage.
        result = aggregate_accounting([dict(run_id=e["accounting_id"], token_accounting=e["accounting"]) for e in values])
        result["accounting_units"] = result.pop("run_count")
        for currency, costs in result["rate_card_estimate_by_currency"].items():
            estimates = [e["accounting"]["rate_card_estimate"] for e in values
                         if (e["accounting"]["rate_card_estimate"]["currency"] or "unknown") == currency]
            costs["known_component_subtotal"] = format(sum((Decimal(e.get("known_subtotal", "0")) for e in estimates), Decimal(0)), "f")
            costs["unknown_cache_write_units"] = sum(e["cache_write_input"] is None for e in estimates)
        return result
    return dict(total=summarize(entries), by_model_source=[
        dict(model=key[0], provider=key[1], source=key[2], **summarize(values)) for key, values in groups.items()],
        billing_scope="Reference rate estimates; reasoning included in output; unknown components are not free.")
