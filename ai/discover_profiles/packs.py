from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field

from .models import get_sbert_model
from .helpers import _mmr


class PackCandidate(BaseModel):
    interests: List[str] = Field(default_factory=list)
    userCount: int = 0
    generatedAt: Optional[str] = None  # ISO8601 timestamp


class PackRankingRequest(BaseModel):
    userInterests: List[str] = Field(default_factory=list)
    packs: List[PackCandidate] = Field(default_factory=list)
    topK: Optional[int] = None
    diversityWeight: float = 0.15  # 0..0.4, higher -> more diversity


class RankedPack(BaseModel):
    interests: List[str] = Field(default_factory=list)
    score: float
    reasons: List[str] = Field(default_factory=list)


_CATEGORY_MAPPING: Dict[str, List[str]] = {
    "Creative": ["Photography", "Art", "Design", "Writing", "Music", "Dancing"],
    "Outdoor": ["Hiking", "Camping", "Running", "Cycling", "Swimming", "Sports"],
    "Food & Drink": ["Coffee", "Cooking", "Wine", "Craft Beer", "Restaurants", "Food"],
    "Technology": ["Gaming", "Programming", "AI", "Gadgets", "Tech"],
    "Lifestyle": ["Fitness", "Yoga", "Meditation", "Travel", "Reading"],
}


def _category_of(interest: str) -> str:
    name = (interest or "").strip().lower()
    for cat, words in _CATEGORY_MAPPING.items():
        for w in words:
            if w.lower() in name:
                return cat
    return "Other"


def _popularity_score(user_count: int) -> float:
    # Smooth, monotonic, 0..1
    x = max(0.0, float(user_count))
    return float(np.tanh(np.log1p(x) / 4.0))


def _freshness_score(generated_at_iso: Optional[str]) -> float:
    if not generated_at_iso:
        return 0.5
    try:
        t = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00"))
    except Exception:
        return 0.5
    now = datetime.now(timezone.utc)
    age_hours = max(0.0, (now - t.astimezone(timezone.utc)).total_seconds() / 3600.0)
    # 0h -> ~1.0, 24h -> ~0.5, 72h -> ~0.2
    return float(np.exp(-age_hours / 36.0))


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = get_sbert_model()
    return np.asarray(model.encode(texts), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return num / den


def rank_packs(request: PackRankingRequest) -> List[RankedPack]:
    if not request.packs:
        return []

    # Prepare embeddings
    user_interest_text = ", ".join([s.strip() for s in request.userInterests or []]) or ""
    user_embed = _embed_texts([user_interest_text])[0]
    pack_texts = [", ".join(p.interests) for p in request.packs]
    pack_embeds = _embed_texts(pack_texts)

    # Base scores per pack
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(request.packs):
        interests = p.interests or []
        if not interests:
            continue
            
        # Calculate category matches
        pack_categories = {_category_of(interest) for interest in interests}
        user_categories = {_category_of(ui) for ui in (request.userInterests or [])}
        cat_match = len(pack_categories & user_categories) / max(1, len(pack_categories))
        
        sem_sim = _cosine(user_embed, pack_embeds[i]) if user_interest_text else 0.0
        pop = _popularity_score(p.userCount or 0)
        fresh = _freshness_score(p.generatedAt)

        # Composite score (tuned weights)
        score = (
            0.50 * sem_sim +
            0.20 * cat_match +
            0.20 * pop +
            0.10 * fresh
        )

        reasons: List[str] = []
        if sem_sim >= 0.35:
            reasons.append("matches your interests")
        if cat_match >= 0.5:
            reasons.append("same vibe category")
        if pop >= 0.6:
            reasons.append("popular now")
        if fresh >= 0.6:
            reasons.append("freshly curated")

        rows.append({
            "interests": interests,
            "score": float(score),
            "reasons": reasons[:2],
        })

    # Diversity with MMR
    base_scores = np.array([r["score"] for r in rows], dtype=np.float32)
    lam = float(1.0 - max(0.0, min(0.4, request.diversityWeight)))
    topk = min(len(rows), request.topK or len(rows))
    order = _mmr(base_scores, pack_embeds, lam, topk=topk)

    return [RankedPack(**rows[i]) for i in order]


# ------------------- Name generation -------------------

_CATEGORY_NAME_SEEDS: Dict[str, List[str]] = {
    "Creative": ["Muse Crew", "Art Mingle", "Inspo Squad", "Pixel Pals", "Beat Bunch"],
    "Outdoor": ["Trail Tribe", "Sun Seekers", "Peak Pals", "Camp Crew", "Ride Squad"],
    "Food & Drink": ["Snack Squad", "Sip Circle", "Bite Buds", "Flavor Fam", "Cafe Crew"],
    "Technology": ["Code Crew", "Tech Tribe", "Gadget Gang", "AI Allies", "Pixel Pack"],
    "Lifestyle": ["Zen Squad", "Fit Fam", "Book Buds", "Travel Tribe", "Vibe Crew"],
    "Other": ["Vibe Crew", "Fun Squad", "Good Times", "Chill Circle", "Mingle Pack"],
}


def _truncate_to_2_or_3_words(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if not words:
        return "Fun Squad"
    if len(words) <= 3:
        return " ".join(words)
    return " ".join(words[:3])


def generate_fun_pack_name(interest: str, category: Optional[str] = None, model=None) -> str:
    cat = (category or _category_of(interest)) or "Other"
    # Try LLM if available
    if model is not None:
        try:
            prompt = (
                "You name interest-based social packs. Return ONE playful name, 2-3 words max, "
                "no emojis, title case, no quotes. It should be based on the interest.\n"
                f"Interest: {interest}\n"
                f"Category: {cat}\n"
            )
            resp = model.invoke(prompt)
            if hasattr(resp, "content"):
                name = str(resp.content).strip()
            else:
                name = str(resp).strip()
            return _truncate_to_2_or_3_words(name)
        except Exception:
            pass

    # Fallback: pick from seeded list for category, tweak with interest keyword if useful
    seeds = _CATEGORY_NAME_SEEDS.get(cat, _CATEGORY_NAME_SEEDS["Other"]) or ["Fun Squad"]
    base = seeds[hash(interest) % len(seeds)]
    return _truncate_to_2_or_3_words(base)



