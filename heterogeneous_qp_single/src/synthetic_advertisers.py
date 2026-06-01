"""Advertiser profiles and segment bid matrix."""

from __future__ import annotations


def get_advertisers() -> list[dict]:
    advertisers = [
        {
            "advertiser_id": "adv_00",
            "name": "BudgetJet",
            "category": "airline",
            "target_audience": "budget",
            "price_level": "low",
            "ad_text": (
                "BudgetJet offers no-frills flights to Hawaii with low base fares, "
                "flexible carry-on options, and fare alerts for travelers who want "
                "to spend more on beaches and food than on airfare."
            ),
            "relevance_notes": "Budget airline for price-sensitive Hawaii travelers.",
        },
        {
            "advertiser_id": "adv_01",
            "name": "AeroLux",
            "category": "airline",
            "target_audience": "luxury",
            "price_level": "high",
            "ad_text": (
                "AeroLux provides lie-flat premium cabins, private lounge access, "
                "concierge itinerary support, and seamless flights for travelers "
                "who want Hawaii to feel refined from departure to arrival."
            ),
            "relevance_notes": "Premium airline for affluent travelers.",
        },
        {
            "advertiser_id": "adv_02",
            "name": "CozyStay",
            "category": "lodging",
            "target_audience": "budget",
            "price_level": "low",
            "ad_text": (
                "CozyStay helps Hawaii visitors find clean hostels, simple guesthouses, "
                "and budget vacation rentals near beaches, transit, and local food spots."
            ),
            "relevance_notes": "Affordable lodging for budget travelers.",
        },
        {
            "advertiser_id": "adv_03",
            "name": "GrandCarlton",
            "category": "lodging",
            "target_audience": "luxury",
            "price_level": "high",
            "ad_text": (
                "GrandCarlton curates oceanfront suites, spa retreats, private cabanas, "
                "and concierge-led Hawaii experiences for travelers seeking a polished "
                "five-star stay."
            ),
            "relevance_notes": "Luxury lodging and resort experience.",
        },
        {
            "advertiser_id": "adv_04",
            "name": "DealMart Travel",
            "category": "marketplace",
            "target_audience": "budget",
            "price_level": "low",
            "ad_text": (
                "DealMart Travel bundles discounted flights, sunscreen, luggage, snacks, "
                "and local activity coupons so practical travelers can plan Hawaii without "
                "overstretching their budget."
            ),
            "relevance_notes": "Budget marketplace for travel essentials.",
        },
        {
            "advertiser_id": "adv_05",
            "name": "BloomHouse",
            "category": "clothing",
            "target_audience": "luxury",
            "price_level": "high",
            "ad_text": (
                "BloomHouse offers designer resort wear, silk dresses, tailored linen, "
                "and personal styling for travelers who want an elegant Hawaii wardrobe."
            ),
            "relevance_notes": "Luxury clothing and styling.",
        },
        {
            "advertiser_id": "adv_06",
            "name": "FastFit",
            "category": "clothing",
            "target_audience": "budget",
            "price_level": "low",
            "ad_text": (
                "FastFit sells affordable swimsuits, sandals, sundresses, and wrinkle-free "
                "travel outfits for Hawaii visitors comparing practical looks under a tight budget."
            ),
            "relevance_notes": "Low-cost clothing for price-sensitive users.",
        },
        {
            "advertiser_id": "adv_07",
            "name": "SunGuard Essentials",
            "category": "travel_goods",
            "target_audience": "both",
            "price_level": "medium",
            "ad_text": (
                "SunGuard Essentials makes reef-safe sunscreen, compact hats, beach bags, "
                "and after-sun care kits for any Hawaii itinerary."
            ),
            "relevance_notes": "Broadly relevant Hawaii travel goods.",
        },
        {
            "advertiser_id": "adv_08",
            "name": "WanderBite",
            "category": "food_local_discovery",
            "target_audience": "both",
            "price_level": "medium",
            "ad_text": (
                "WanderBite recommends local poke shops, farmers markets, tasting menus, "
                "and hidden neighborhood restaurants for travelers exploring Hawaii through food."
            ),
            "relevance_notes": "Food discovery relevant to both groups.",
        },
        {
            "advertiser_id": "adv_09",
            "name": "MusicStream",
            "category": "irrelevant",
            "target_audience": "neither",
            "price_level": "medium",
            "ad_text": (
                "MusicStream gives listeners playlists, offline listening, and personalized "
                "music recommendations across devices."
            ),
            "relevance_notes": "Control advertiser with weak relevance to Hawaii travel.",
        },
        {
            "advertiser_id": "adv_10",
            "name": "BrainChips",
            "category": "irrelevant",
            "target_audience": "neither",
            "price_level": "medium",
            "ad_text": (
                "BrainChips develops processors for laptops, servers, and cloud workloads "
                "with an emphasis on reliability and performance."
            ),
            "relevance_notes": "Control advertiser unrelated to travel planning.",
        },
    ]
    for advertiser in advertisers:
        advertiser["profile_text"] = (
            f"{advertiser['name']}. Category: {advertiser['category']}. "
            f"Target: {advertiser['target_audience']}. Price level: {advertiser['price_level']}. "
            f"{advertiser['ad_text']}"
        )
    return advertisers


def get_bids_by_segment(advertisers: list[dict]) -> dict:
    bids = {
        "BudgetJet": {"budget": 3.0, "luxury": 0.5},
        "AeroLux": {"budget": 0.5, "luxury": 3.0},
        "CozyStay": {"budget": 2.5, "luxury": 0.5},
        "GrandCarlton": {"budget": 0.0, "luxury": 3.0},
        "DealMart Travel": {"budget": 2.0, "luxury": 1.0},
        "BloomHouse": {"budget": 0.5, "luxury": 2.5},
        "FastFit": {"budget": 1.5, "luxury": 0.0},
        "SunGuard Essentials": {"budget": 1.2, "luxury": 1.2},
        "WanderBite": {"budget": 1.5, "luxury": 1.5},
        "MusicStream": {"budget": 0.2, "luxury": 0.2},
        "BrainChips": {"budget": 0.1, "luxury": 0.1},
    }
    return {
        advertiser["advertiser_id"]: {
            "name": advertiser["name"],
            "budget": bids[advertiser["name"]]["budget"],
            "luxury": bids[advertiser["name"]]["luxury"],
        }
        for advertiser in advertisers
    }

