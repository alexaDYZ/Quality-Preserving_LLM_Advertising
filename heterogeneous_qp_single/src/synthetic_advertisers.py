"""Advertiser profiles and segment bid matrix."""

from __future__ import annotations


HAN_DAI_SCENARIO_1_BIDS = {
    "SunWing Airlines": 3.0,
    "TropicStay": 3.0,
    "WanderBite": 2.0,
    "NovaSkin": 2.0,
    "GridPower Bank": 1.0,
}

HAN_DAI_SCENARIO_1_RELEVANCE = {
    "SunWing Airlines": 0.62,
    "TropicStay": 0.67,
    "WanderBite": 0.61,
    "NovaSkin": 0.49,
    "GridPower Bank": 0.59,
}

SCENARIO_1_TARGET_ASSIGNMENT_SEED = 211
SCENARIO_1_TARGET_AUDIENCE = {
    "SunWing Airlines": "luxury",
    "TropicStay": "budget",
    "WanderBite": "budget",
    "NovaSkin": "luxury",
    "GridPower Bank": "luxury",
}


def get_advertisers(scenario: str = "scenario_1") -> list[dict]:
    if scenario == "scenario_1":
        return _with_profile_text(_scenario_1_advertisers())
    if scenario == "current":
        return _with_profile_text(_current_advertisers())
    raise ValueError(f"Unknown advertiser scenario: {scenario!r}")


def _scenario_1_advertisers() -> list[dict]:
    return [
        {
            "advertiser_id": "adv_00",
            "name": "SunWing Airlines",
            "paper_name": "SunWing",
            "category": "airline",
            "target_audience": SCENARIO_1_TARGET_AUDIENCE["SunWing Airlines"],
            "price_level": "high",
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS["SunWing Airlines"],
            "han_dai_q_i_v1": HAN_DAI_SCENARIO_1_RELEVANCE["SunWing Airlines"],
            "target_assignment_note": (
                "Scenario 1 randomized target assignment with seed "
                f"{SCENARIO_1_TARGET_ASSIGNMENT_SEED}."
            ),
            "ad_text": (
                "Take off with SunWing Airlines, your trusted partner for seamless "
                "travel across the Pacific and beyond. With spacious cabins, "
                "award-winning in-flight service, and direct routes to the world's "
                "most breathtaking destinations, SunWing makes every journey as "
                "memorable as the destination itself. Book today and enjoy exclusive "
                "deals, flexible cancellation, and a loyalty rewards program that "
                "turns miles into memories. Your next adventure starts here - fly "
                "with SunWing."
            ),
            "relevance_notes": (
                "Original QP single scenario 1 airline advertiser; content unchanged, "
                "with randomized luxury target label for heterogeneity."
            ),
        },
        {
            "advertiser_id": "adv_01",
            "name": "TropicStay",
            "paper_name": "TropicStay",
            "category": "lodging",
            "target_audience": SCENARIO_1_TARGET_AUDIENCE["TropicStay"],
            "price_level": "low",
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS["TropicStay"],
            "han_dai_q_i_v1": HAN_DAI_SCENARIO_1_RELEVANCE["TropicStay"],
            "target_assignment_note": (
                "Scenario 1 randomized target assignment with seed "
                f"{SCENARIO_1_TARGET_ASSIGNMENT_SEED}."
            ),
            "ad_text": (
                "Discover the perfect retreat with TropicStay, the premier platform "
                "for booking vacation rentals in the world's most beautiful "
                "destinations. From oceanfront villas to cozy jungle hideaways, "
                "TropicStay connects travelers with handpicked, locally-owned "
                "properties that offer authentic experiences you won't find anywhere "
                "else. With instant booking, 24/7 guest support, and verified reviews "
                "from real travelers, planning your dream getaway has never been "
                "easier. Explore, book, and belong - only with TropicStay."
            ),
            "relevance_notes": (
                "Original QP single scenario 1 lodging advertiser; content unchanged, "
                "with randomized budget target label for heterogeneity."
            ),
        },
        {
            "advertiser_id": "adv_02",
            "name": "WanderBite",
            "paper_name": "WanderBite",
            "category": "food_local_discovery",
            "target_audience": SCENARIO_1_TARGET_AUDIENCE["WanderBite"],
            "price_level": "low",
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS["WanderBite"],
            "han_dai_q_i_v1": HAN_DAI_SCENARIO_1_RELEVANCE["WanderBite"],
            "target_assignment_note": (
                "Scenario 1 randomized target assignment with seed "
                f"{SCENARIO_1_TARGET_ASSIGNMENT_SEED}."
            ),
            "ad_text": (
                "Unlock the culinary soul of every destination with WanderBite, the "
                "app that connects food lovers to the best local restaurants, street "
                "food spots, and hidden dining gems around the globe. Whether you're "
                "craving fresh poke bowls on the North Shore or a Michelin-starred "
                "tasting menu in the city, WanderBite guides you there with curated "
                "recommendations, real-time reservations, and exclusive foodie deals. "
                "Because the best travel memories are made at the table. Download "
                "WanderBite and taste the world."
            ),
            "relevance_notes": (
                "Original QP single scenario 1 food discovery advertiser; content "
                "unchanged, with randomized budget target label for heterogeneity."
            ),
        },
        {
            "advertiser_id": "adv_03",
            "name": "NovaSkin",
            "paper_name": "NovaSkin",
            "category": "skincare",
            "target_audience": SCENARIO_1_TARGET_AUDIENCE["NovaSkin"],
            "price_level": "high",
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS["NovaSkin"],
            "han_dai_q_i_v1": HAN_DAI_SCENARIO_1_RELEVANCE["NovaSkin"],
            "target_assignment_note": (
                "Scenario 1 randomized target assignment with seed "
                f"{SCENARIO_1_TARGET_ASSIGNMENT_SEED}."
            ),
            "ad_text": (
                "Introducing NovaSkin, the dermatologist-approved skincare line "
                "engineered for the modern lifestyle. From SPF-50 daily moisturizers "
                "to overnight repair serums, NovaSkin's lightweight, reef-safe "
                "formulas protect and restore your skin whether you're under the "
                "office lights or the open sun. Trusted by over 10 million customers "
                "worldwide, NovaSkin combines cutting-edge biotechnology with clean, "
                "sustainable ingredients to deliver visible results you can feel "
                "confident about. Because great skin doesn't take a vacation - and "
                "neither does NovaSkin."
            ),
            "relevance_notes": (
                "Original QP single scenario 1 skincare advertiser; content unchanged, "
                "with randomized luxury target label for heterogeneity."
            ),
        },
        {
            "advertiser_id": "adv_04",
            "name": "GridPower Bank",
            "paper_name": "GridPower",
            "category": "travel_goods",
            "target_audience": SCENARIO_1_TARGET_AUDIENCE["GridPower Bank"],
            "price_level": "high",
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS["GridPower Bank"],
            "han_dai_q_i_v1": HAN_DAI_SCENARIO_1_RELEVANCE["GridPower Bank"],
            "target_assignment_note": (
                "Scenario 1 randomized target assignment with seed "
                f"{SCENARIO_1_TARGET_ASSIGNMENT_SEED}."
            ),
            "ad_text": (
                "Stay charged through every adventure with GridPower Bank, the "
                "ultra-slim, high-capacity portable charger built for life on the "
                "move. Featuring rapid-charge technology, dual USB-C ports, and a "
                "rugged waterproof design, GridPower Bank keeps your devices powered "
                "through long flights, beach days, and everything in between. Compact "
                "enough to fit in your pocket, powerful enough to charge your laptop "
                "twice over - GridPower Bank is the travel essential you didn't know "
                "you needed. Power up and go."
            ),
            "relevance_notes": (
                "Original QP single scenario 1 portable power advertiser; content "
                "unchanged, with randomized luxury target label for heterogeneity."
            ),
        },
    ]


def _current_advertisers() -> list[dict]:
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
    return advertisers


def _with_profile_text(advertisers: list[dict]) -> list[dict]:
    for advertiser in advertisers:
        advertiser["profile_text"] = (
            f"{advertiser['name']}. Category: {advertiser['category']}. "
            f"Target: {advertiser['target_audience']}. Price level: {advertiser['price_level']}. "
            f"{advertiser['ad_text']}"
        )
    return advertisers


def get_bids_by_segment(advertisers: list[dict], scenario: str = "scenario_1") -> dict:
    if scenario == "scenario_1":
        return _scenario_1_bids_by_segment(advertisers)
    if scenario == "current":
        return _current_bids_by_segment(advertisers)
    raise ValueError(f"Unknown advertiser scenario: {scenario!r}")


def _scenario_1_bids_by_segment(advertisers: list[dict]) -> dict:
    missing = [
        advertiser["name"]
        for advertiser in advertisers
        if advertiser["name"] not in HAN_DAI_SCENARIO_1_BIDS
    ]
    if missing:
        raise ValueError(f"Scenario 1 is missing Han-Dai bids for: {missing}")

    return {
        advertiser["advertiser_id"]: {
            "name": advertiser["name"],
            "han_dai_bid": HAN_DAI_SCENARIO_1_BIDS[advertiser["name"]],
            "budget": HAN_DAI_SCENARIO_1_BIDS[advertiser["name"]] - 0.5,
            "luxury": HAN_DAI_SCENARIO_1_BIDS[advertiser["name"]] + 0.5,
            "expected_bid_under_50_50_segments": HAN_DAI_SCENARIO_1_BIDS[advertiser["name"]],
            "bid_design_note": (
                "Scenario 1 uses original_bid +/- 0.5 with 50 budget users and "
                "50 luxury users, so the expected bid equals Han-Dai's scalar bid."
            ),
        }
        for advertiser in advertisers
    }


def _current_bids_by_segment(advertisers: list[dict]) -> dict:
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
