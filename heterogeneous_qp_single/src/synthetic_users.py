"""Synthetic LLM memories and affordability classification."""

from __future__ import annotations

import math
import random
import re
from typing import List


LUXURY_SIGNALS = {
    "google": 1.0,
    "startup exit": 1.2,
    "equity": 0.7,
    "private": 1.0,
    "concierge": 0.9,
    "five-star": 1.0,
    "suite": 0.7,
    "business class": 1.0,
    "first class": 1.1,
    "switzerland": 0.6,
    "maldives": 0.7,
    "santorini": 0.5,
    "diamond": 1.0,
    "designer": 0.8,
    "michelin": 0.8,
    "ritz": 1.0,
    "investment banker": 1.0,
    "partner at a law firm": 1.0,
    "annual bonus": 0.8,
}

BUDGET_SIGNALS = {
    "front desk": 1.0,
    "clinic": 0.8,
    "shift": 0.7,
    "overtime": 0.7,
    "coupon": 0.9,
    "discount": 0.8,
    "$20": 1.0,
    "$30": 0.8,
    "student loan": 0.9,
    "rent": 0.6,
    "hostel": 0.7,
    "public transit": 0.6,
    "spirit": 0.8,
    "sale rack": 0.8,
    "paycheck": 0.7,
    "shared room": 0.7,
    "cashback": 0.6,
    "budget": 0.6,
}


def generate_user_memories(n_users: int = 100, seed: int = 211) -> list[dict]:
    rng = random.Random(seed)
    n_budget = n_users // 2
    archetypes = ["budget"] * n_budget + ["luxury"] * (n_users - n_budget)
    rng.shuffle(archetypes)

    users = []
    for idx, archetype in enumerate(archetypes):
        memory_text = (
            _budget_memory(rng, idx)
            if archetype == "budget"
            else _luxury_memory(rng, idx)
        )
        users.append(
            {
                "user_id": f"user_{idx:03d}",
                "memory_text": memory_text,
                "generation_archetype": archetype,
            }
        )
    return users


def classify_users(raw_users: list[dict], embeddings: List[List[float]]) -> list[dict]:
    classified = []
    for user, embedding in zip(raw_users, embeddings):
        budget_raw, budget_matches = _weighted_signal_score(user["memory_text"], BUDGET_SIGNALS)
        luxury_raw, luxury_matches = _weighted_signal_score(user["memory_text"], LUXURY_SIGNALS)
        luxury_prob = _sigmoid((luxury_raw - budget_raw) / 1.75)
        budget_score = round(1.0 - luxury_prob, 6)
        luxury_score = round(luxury_prob, 6)
        segment = "luxury" if luxury_score >= budget_score else "budget"
        explanation = (
            f"Classified as {segment}: luxury signals={luxury_matches[:4]}, "
            f"budget signals={budget_matches[:4]}."
        )
        classified.append(
            {
                "user_id": user["user_id"],
                "memory_text": user["memory_text"],
                "generation_archetype": user["generation_archetype"],
                "embedding": [round(v, 8) for v in embedding],
                "budget_score": budget_score,
                "luxury_score": luxury_score,
                "segment": segment,
                "classifier_explanation": explanation,
            }
        )
    return classified


def _weighted_signal_score(text: str, signals: dict[str, float]) -> tuple[float, list[str]]:
    lowered = text.lower()
    score = 0.0
    matches = []
    for phrase, weight in signals.items():
        if re.search(re.escape(phrase), lowered):
            score += weight
            matches.append(phrase)
    return score, matches


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _budget_memory(rng: random.Random, idx: int) -> str:
    jobs = [
        "works at the front desk of a clinic",
        "is a medical assistant picking up extra shifts",
        "is a public school aide finishing evening classes",
        "works retail and tracks each paycheck carefully",
        "is a graduate student with student loan payments",
        "coordinates appointments at a dental office",
    ]
    constraints = [
        "often asks the LLM to compare a $20 dress with a $30 dress before buying",
        "saves links to coupon codes and cashback offers",
        "asks whether Spirit or a basic economy fare is worth the inconvenience",
        "prefers public transit, shared rooms, and free museum days",
        "keeps a monthly rent spreadsheet and asks for grocery discount ideas",
        "looks for sale rack outfits that still look polished at work",
    ]
    travel = [
        "asks for Hawaii plans built around beaches, bus routes, and affordable food trucks",
        "dreams about Switzerland but asks for hostel routes and shoulder-season dates",
        "compares low-cost Caribbean trips with staycations near family",
        "asks for a three-day itinerary that avoids expensive resort fees",
        "wants snorkeling, hikes, and farmers markets more than spa appointments",
    ]
    overlap = [
        "occasionally asks about one nice hotel night if there is a major discount",
        "likes elegant outfits but filters recommendations by price first",
        "mentions wanting a special dinner, then asks how to keep the rest of the trip cheap",
        "saves luxury resort photos as inspiration while planning practical trips",
    ]
    return " ".join(
        [
            f"Memory profile {idx}: The user {rng.choice(jobs)}.",
            rng.choice(constraints) + ".",
            rng.choice(travel) + ".",
            rng.choice(overlap) + ".",
            "The user's LLM questions imply careful budgeting, time pressure, and high price sensitivity.",
        ]
    )


def _luxury_memory(rng: random.Random, idx: int) -> str:
    jobs = [
        "used to work at Google and now has substantial equity savings",
        "is an investment banker with a large annual bonus",
        "is a partner at a law firm and delegates most travel planning",
        "sold a startup exit and asks the LLM to coordinate family travel",
        "runs a venture-backed company and travels frequently for conferences",
        "is a senior product executive who values convenience over price",
    ]
    spending = [
        "bought a fiancee a 10 carat diamond and asks about insurance appraisals",
        "asks whether first class or business class is better for overnight flights",
        "keeps notes on designer resort wear and private shopping appointments",
        "asks for Michelin restaurants, spa suites, and concierge transfers",
        "compares Ritz-style resorts with private villas in Switzerland and the Maldives",
        "asks the LLM to shortlist oceanfront suites with private cabanas",
    ]
    travel = [
        "often asks for vacation recommendations in Switzerland, Santorini, and Hawaii",
        "prefers five-star hotels, private guides, and flexible cancellation",
        "asks for quiet luxury, premium lounge access, and low-friction logistics",
        "wants Hawaii plans with helicopter tours, chef-led dinners, and spa days",
        "asks for resort neighborhoods where privacy and service quality matter most",
    ]
    overlap = [
        "still asks about points redemptions when they do not reduce comfort",
        "likes a good deal but does not want to trade down on convenience",
        "occasionally asks whether a practical sunscreen or luggage item is worth it",
        "enjoys local food trucks too, but usually pairs them with a luxury hotel stay",
    ]
    return " ".join(
        [
            f"Memory profile {idx}: The user {rng.choice(jobs)}.",
            rng.choice(spending) + ".",
            rng.choice(travel) + ".",
            rng.choice(overlap) + ".",
            "The user's LLM questions imply high affordability, convenience seeking, and low price sensitivity.",
        ]
    )

