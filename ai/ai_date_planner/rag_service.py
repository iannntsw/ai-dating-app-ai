from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .data_processor import Location
from .embedding_service import EmbeddingService
from .rule_engine import UserPreferences, FilterResult
import numpy as np

@dataclass
class RAGResult:
    """Result of RAG-based location retrieval"""
    relevant_locations: List[Location]
    relevance_scores: Dict[str, float]  # Location ID -> relevance score
    query_embedding: np.ndarray
    search_stats: Dict[str, Any]

class RAGService:
    """
    Retrieval-Augmented Generation service for finding relevant locations.
    
    This service takes filtered locations and user preferences to find the most
    relevant locations using semantic similarity search.
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """Initialize RAG service with embedding service"""
        self.embedding_service = embedding_service
        self.max_results = 70  # Maximum locations to return (more focused results)
    
    def find_relevant_locations(self, filter_result: FilterResult, preferences: UserPreferences) -> RAGResult:
        """
        Find the most relevant locations using semantic similarity search.
        
        Args:
            filter_result: Result from rule-based filtering
            preferences: User preferences
            
        Returns:
            RAGResult with relevant locations and scores
        """
        print(f"Starting RAG-based location retrieval with {len(filter_result.filtered_locations)} filtered locations...")
        
        # Generate query embedding
        query_text = self._build_query_text(preferences)
        query_embedding = self.embedding_service.generate_embedding(query_text)
        
        print(f"Generated query embedding for: '{query_text[:100]}...'")
        
        # Try FAISS accelerated search first (graceful fallback to cosine loop)
        relevance_scores = {}
        use_faiss = False
        try:
            # Ensure embeddings and FAISS index are ready
            self.embedding_service.ensure_index_ready()
            use_faiss = self.embedding_service.index is not None
        except FileNotFoundError:
            use_faiss = False

        if use_faiss:
            # Use FAISS to get top similar locations by query, then intersect with filtered set
            faiss_results = self.embedding_service.similarity_search(query_text, k=200)
            faiss_ids = {res['location'].id: res['score'] for res in faiss_results}

            # Keep only those present in rule-filtered locations
            filtered_ids = {loc.id for loc in filter_result.filtered_locations}
            for loc_id, score in faiss_ids.items():
                if loc_id in filtered_ids:
                    relevance_scores[loc_id] = float(score)

            # If intersection is too small, fall back to cosine for the filtered set
            if len(relevance_scores) < 5:
                relevance_scores = self._calculate_relevance_scores(
                    filter_result.filtered_locations,
                    query_embedding,
                    filter_result.location_scores
                )
        else:
            # Calculate relevance scores for all filtered locations (cosine similarity)
            relevance_scores = self._calculate_relevance_scores(
                filter_result.filtered_locations, 
                query_embedding,
                filter_result.location_scores
            )
        
        # Sort by combined relevance and proximity scores
        sorted_locations = self._rank_locations(
            filter_result.filtered_locations,
            relevance_scores,
            filter_result.location_scores
        )
        
        # DIVERSITY SAMPLING: Ensure mix of location types, not just food
        top_locations = self._sample_diverse_locations(sorted_locations, self.max_results)
        
        print(f"RAG retrieval complete: {len(top_locations)} most relevant locations selected")
        
        return RAGResult(
            relevant_locations=top_locations,
            relevance_scores=relevance_scores,
            query_embedding=query_embedding,
            search_stats={
                'total_filtered': len(filter_result.filtered_locations),
                'top_results': len(top_locations),
                'query_text': query_text
            }
        )
    
    def _build_query_text(self, preferences: UserPreferences) -> str:
        """Build a comprehensive query text from preferences"""
        query_parts = []
        
        # Add time-based context
        time_context = self._get_time_context(preferences)
        if time_context:
            query_parts.append(time_context)
        
        # Add interest-based context
        interest_context = self._get_interest_context(preferences.interests)
        if interest_context:
            query_parts.append(interest_context)
        
        # Add date type context
        date_type_context = self._get_date_type_context(preferences.date_type)
        if date_type_context:
            query_parts.append(date_type_context)
        
        # Add budget context
        budget_context = self._get_budget_context(preferences.budget_tier)
        if budget_context:
            query_parts.append(budget_context)
        
        return " ".join(query_parts)
    
    def _get_time_context(self, preferences: UserPreferences) -> str:
        """Generate time-based context for the query"""
        duration = preferences.get_duration_hours()
        
        if preferences.time_of_day == "morning":
            return f"morning activities for {duration:.1f} hours, breakfast and brunch options"
        elif preferences.time_of_day == "afternoon":
            return f"afternoon activities for {duration:.1f} hours, lunch and daytime attractions"
        elif preferences.time_of_day == "evening":
            return f"evening activities for {duration:.1f} hours, dinner and sunset views"
        else:  # night
            return f"night activities for {duration:.1f} hours, late night dining and entertainment"
    
    def _get_interest_context(self, interests: List[str]) -> str:
        """Generate interest-based context for the query"""
        if not interests:
            return ""
        
        interest_descriptions = {
            "food": "restaurants, cafes, food markets, local cuisine",
            "culture": "museums, galleries, cultural sites, heritage locations",
            "nature": "parks, gardens, outdoor spaces, scenic views",
            "sports": "sports facilities, fitness centers, active activities",
            "art": "art galleries, exhibitions, creative spaces, artistic venues",
            "shopping": "shopping malls, markets, boutiques, retail areas"
        }
        
        descriptions = [interest_descriptions.get(interest, interest) for interest in interests]
        return f"interested in {', '.join(descriptions)}"
    
    def _get_date_type_context(self, date_type: str) -> str:
        """Generate date type context for the query with enhanced semantic matching"""
        date_type_descriptions = {
            "casual": "casual and relaxed atmosphere, comfortable settings, laid-back vibe, easy-going environment, friendly venues, bistro dining, relaxed cafes, comfortable restaurants",
            "romantic": "romantic and intimate atmosphere, cozy ambiance, candlelit dining, scenic views, sunset locations, private spaces, elegant restaurants, beautiful setting, date night vibes, couple-friendly, charming atmosphere, fine dining, rooftop dining, waterfront views",
            "adventurous": "adventure and outdoor activities, exciting experiences, thrilling activities, active sports, unique experiences, unconventional dining, street food adventures, outdoor seating, rooftop venues, unique cuisines, fusion restaurants, experimental menus",
            "cultural": "cultural and educational experiences, historical significance, museums, heritage sites, art galleries, traditional venues, authentic local cuisine, cultural dining, heritage restaurants, traditional ambiance, peranakan food, historical settings"
        }
        
        return date_type_descriptions.get(date_type, f"{date_type} atmosphere")
    
    def _get_budget_context(self, budget_tier: str) -> str:
        """Generate budget context for the query"""
        budget_descriptions = {
            "$": "budget-friendly, affordable, cheap options",
            "$$": "moderate pricing, mid-range, casual dining",
            "$$$": "upscale, fine dining, premium experiences",
            "$$$$": "high-end, luxury, exclusive venues"
        }
        
        return budget_descriptions.get(budget_tier, f"{budget_tier} budget range")
    
    def _calculate_relevance_scores(self, locations: List[Location], query_embedding: np.ndarray, proximity_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate semantic relevance scores for locations"""
        relevance_scores = {}
        
        # Ensure embeddings are loaded (loads once for all locations)
        try:
            self.embedding_service.load_embeddings()
        except FileNotFoundError:
            print("⚠️ No embeddings found, falling back to proximity scores only")
            return {loc.id: proximity_scores.get(loc.id, 0.0) for loc in locations}
        
        for location in locations:
            # Get location embedding (now uses pre-loaded embeddings)
            location_embedding = self.embedding_service.get_location_embedding(location.id)
            
            if location_embedding is not None:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, location_embedding)
                relevance_scores[location.id] = float(similarity)
            else:
                # Fallback to proximity score if no embedding
                relevance_scores[location.id] = proximity_scores.get(location.id, 0.0)
        
        return relevance_scores
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _rank_locations(self, locations: List[Location], relevance_scores: Dict[str, float], proximity_scores: Dict[str, float]) -> List[Location]:
        """Rank locations by combined relevance and proximity scores"""
        # Combine relevance (60%) and proximity (40%) scores - balance semantic match with proximity
        combined_scores = {}
        
        for location in locations:
            relevance = relevance_scores.get(location.id, 0.0)
            proximity = proximity_scores.get(location.id, 0.0)
            
            # Weighted combination: 60% semantic relevance, 40% proximity (prioritize proximity)
            combined_score = 0.6 * relevance + 0.4 * proximity
            combined_scores[location.id] = combined_score
        
        # Sort by combined score (descending)
        sorted_locations = sorted(locations, key=lambda loc: combined_scores[loc.id], reverse=True)
        
        return sorted_locations
    
    def get_rag_summary(self, result: RAGResult) -> str:
        """Get a human-readable summary of RAG results"""
        summary = f"RAG Retrieval Results:\n"
        summary += f"  Total filtered locations: {result.search_stats['total_filtered']}\n"
        summary += f"  Top relevant locations: {result.search_stats['top_results']}\n"
        summary += f"  Query: {result.search_stats['query_text'][:100]}...\n\n"
        
        summary += "Top 5 most relevant locations:\n"
        for i, location in enumerate(result.relevant_locations[:5], 1):
            relevance_score = result.relevance_scores.get(location.id, 0.0)
            summary += f"  {i}. {location.name} (relevance: {relevance_score:.3f})\n"
        
        return summary
    
    def explain_relevance(self, location: Location, query_embedding: np.ndarray) -> str:
        """Explain why a location is relevant to the query"""
        location_embedding = self.embedding_service.get_location_embedding(location.id)
        
        if location_embedding is None:
            return f"No embedding available for {location.name}"
        
        similarity = self._cosine_similarity(query_embedding, location_embedding)
        
        explanation = f"Location: {location.name}\n"
        explanation += f"Type: {location.location_type}\n"
        explanation += f"Description: {location.description[:100]}...\n"
        explanation += f"Relevance Score: {similarity:.3f}\n"
        
        if similarity > 0.8:
            explanation += "Status: Highly relevant to your query"
        elif similarity > 0.6:
            explanation += "Status: Moderately relevant to your query"
        elif similarity > 0.4:
            explanation += "Status: Somewhat relevant to your query"
        else:
            explanation += "Status: Low relevance to your query"
        
        return explanation
    
    def _sample_diverse_locations(self, sorted_locations: List[Location], max_results: int) -> List[Location]:
        """Sample locations with diversity to ensure mix of location types"""
        # Group by type
        by_type = {
            'food': [],
            'attraction': [],
            'activity': [],
            'heritage': []
        }
        
        for loc in sorted_locations:
            if loc.location_type in by_type:
                by_type[loc.location_type].append(loc)
        
        # Calculate allocation (ensure at least some of each type)
        total_non_food = len(by_type['attraction']) + len(by_type['activity']) + len(by_type['heritage'])
        
        if total_non_food == 0:
            # All food, just return top N
            return sorted_locations[:max_results]
        
        # Reserve slots for non-food (at least 30% of results)
        min_non_food = max(30, int(max_results * 0.3))  # At least 30 non-food locations
        max_food = max_results - min_non_food
        
        # Collect results with diversity
        results = []
        
        # Add food locations (up to max_food)
        results.extend(by_type['food'][:max_food])
        
        # Add non-food locations proportionally
        remaining_slots = max_results - len(results)
        
        # Distribute remaining slots across non-food types
        for loc_type in ['attraction', 'activity', 'heritage']:
            type_locs = by_type[loc_type]
            if type_locs and remaining_slots > 0:
                # Take proportional share or all available, whichever is smaller
                take_count = min(len(type_locs), max(5, remaining_slots // 3))  # At least 5 of each
                results.extend(type_locs[:take_count])
                remaining_slots -= take_count
        
        print(f"  Diversity sampling: {len([r for r in results if r.location_type == 'food'])} food, "
              f"{len([r for r in results if r.location_type == 'attraction'])} attractions, "
              f"{len([r for r in results if r.location_type == 'activity'])} activities, "
              f"{len([r for r in results if r.location_type == 'heritage'])} heritage")
        
        return results[:max_results]
