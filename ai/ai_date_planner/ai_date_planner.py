from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .data_processor import DataProcessor, Location
from .embedding_service import EmbeddingService
from .rule_engine import RuleEngine, UserPreferences, FilterResult
from .rag_service import RAGService, RAGResult
import json
import os

@dataclass
class DatePlan:
    """Final date plan with specific locations and timing"""
    itinerary: List[Dict[str, Any]]  # List of activities with times and locations
    total_duration: float
    estimated_cost: str
    summary: str
    alternative_suggestions: List[str]

@dataclass
class DatePlanResult:
    """Complete result of date planning process"""
    date_plan: DatePlan
    filter_result: FilterResult
    rag_result: RAGResult
    processing_stats: Dict[str, Any]

class AIDatePlanner:
    """
    Main AI Date Planner that orchestrates the entire date planning process.
    
    This is the main class that combines:
    1. Data processing (loading and parsing location data)
    2. Rule-based filtering (applying user preferences)
    3. RAG-based retrieval (AI-powered relevance search)
    4. Itinerary generation (creating specific date plans)
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the AI Date Planner with all required services"""
        self.data_dir = data_dir
        
        # Initialize services
        print("Initializing AI Date Planner services...")
        self.data_processor = DataProcessor(data_dir)
        self.embedding_service = EmbeddingService()
        self.rule_engine = RuleEngine()
        self.rag_service = RAGService(self.embedding_service)
        
        # Cache for processed data
        self._locations_cache = None
        self._embeddings_ready = False
        self._current_exclusions = []  # Store current request's exclusions
        
        print("AI Date Planner initialized successfully!")
    
    def plan_date(self, preferences: UserPreferences, exclusions: List[str] = None) -> DatePlanResult:
        """
        Plan a complete date based on user preferences and optional exclusions.
        
        Args:
            preferences: User preferences for the date
            exclusions: Optional list of what user does NOT want (e.g., ["sports", "cultural"])
            
        Returns:
            DatePlanResult with complete date plan and processing details
        """
        print(f"\n🎯 Starting AI Date Planning...")
        print(f"Exclusions: {exclusions or 'None'}")
        print(f"Preferences: {preferences.start_time} - {preferences.end_time or 'flexible'}")
        
        # Validate that starting coordinates are provided
        if preferences.start_latitude is None or preferences.start_longitude is None:
            raise ValueError(
                "Starting location is required. Please select a location in Singapore (e.g., Orchard, Marina Bay, Sentosa, etc.) "
                "before planning your date."
            )
        
        # Store exclusions for use throughout the planning process
        self._current_exclusions = exclusions or []
        
        # Step 1: Load and process location data
        locations = self._get_locations()
        
        # Step 2: Apply rule-based filtering
        print("\n📋 Step 1: Rule-based filtering...")
        filter_result = self.rule_engine.filter_locations(locations, preferences, self._current_exclusions)
        
        # Step 3: RAG-based relevance search
        print("\n🧠 Step 2: AI-powered relevance search...")
        rag_result = self.rag_service.find_relevant_locations(filter_result, preferences)
        
        # Step 4: Generate specific itinerary
        print("\n📅 Step 3: Generating specific itinerary...")
        date_plan = self._generate_itinerary(rag_result, preferences)
        
        # Compile processing statistics
        processing_stats = {
            'total_locations': len(locations),
            'filtered_locations': len(filter_result.filtered_locations),
            'relevant_locations': len(rag_result.relevant_locations),
            'final_activities': len(date_plan.itinerary),
            'embeddings_ready': self._embeddings_ready
        }
        
        print(f"\n✅ Date planning complete!")
        print(f"Final itinerary: {len(date_plan.itinerary)} activities planned")
        
        return DatePlanResult(
            date_plan=date_plan,
            filter_result=filter_result,
            rag_result=rag_result,
            processing_stats=processing_stats
        )
    
    def _get_locations(self) -> List[Location]:
        """Get locations from cache or load from data files"""
        if self._locations_cache is None:
            print("Loading location data...")
            self._locations_cache = self.data_processor.process_all_files()
            print(f"Loaded {len(self._locations_cache)} locations")
        
        return self._locations_cache
    
    def _generate_itinerary(self, rag_result: RAGResult, preferences: UserPreferences) -> DatePlan:
        """Generate a specific itinerary from relevant locations"""
        duration = preferences.get_duration_hours()
        relevant_locations = rag_result.relevant_locations
        
        # Group locations by type
        location_groups = self._group_locations_by_type(relevant_locations)
        
        # Filter food locations for meal appropriateness and date type
        # Note: meal_type-specific filtering happens in _plan_next_meal
        location_groups['food'] = [
            loc for loc in location_groups['food']
            if self._is_appropriate_for_meal(loc) and self._is_appropriate_for_date_type(loc, preferences.date_type)
        ]
        
        # Re-rank food by date type match for ALL date types
        location_groups['food'] = self._rank_by_date_type_match(location_groups['food'], preferences.date_type)
        
        # Generate itinerary based on time of day and duration
        itinerary = self._create_time_based_itinerary(
            location_groups, 
            preferences, 
            duration
        )
        
        # Validate itinerary uses at least 75% of requested time
        total_activity_time = sum(item.get('duration', 0) for item in itinerary)
        time_usage_percentage = (total_activity_time / duration) * 100 if duration > 0 else 0
        
        if time_usage_percentage < 75 and duration > 2 and len(itinerary) > 0:  # Only enforce for dates > 2 hours
            print(f"⚠️ Low coverage: {time_usage_percentage:.1f}% ({total_activity_time:.1f}/{duration:.1f} hours)")
            
            # For very long dates (12+ hours), relax the 75% requirement
            # It's unrealistic to fill 18+ hours continuously
            if duration >= 12.0:
                required_percentage = max(50, 75 - (duration - 12) * 2)  # Gradually reduce from 75% for long dates
                print(f"  📏 Long date ({duration:.1f}h): Relaxing coverage requirement to {required_percentage:.0f}%")
                
                if time_usage_percentage >= required_percentage:
                    print(f"✅ Meets relaxed coverage requirement ({time_usage_percentage:.1f}% >= {required_percentage:.0f}%)")
                    # Skip validation for long dates that meet relaxed requirement
                else:
                    print(f"⚠️ Even relaxed requirement not met ({time_usage_percentage:.1f}% < {required_percentage:.0f}%)")
                    print(f"  Note: For {duration:.1f}-hour dates, continuous activity planning may not be feasible")
            else:
                # For normal dates (< 12 hours), try to extend the last activity to meet 75%
                target_time = duration * 0.75  # 75% of requested duration
                time_needed = target_time - total_activity_time
                
                if time_needed > 0 and len(itinerary) > 0:
                    last_activity = itinerary[-1]
                    
                    # Only extend non-food activities, and cap extension at 3 hours
                    max_extension = 3.0
                    if last_activity.get('type') != 'food' and time_needed <= max_extension:
                        print(f"🔧 Extending last activity by {time_needed:.1f} hours to reach 75% coverage ({target_time:.1f}/{duration:.1f} hours)")
                        last_activity['duration'] += time_needed
                        last_activity['end_time'] = self._add_hours(last_activity['start_time'], last_activity['duration'])
                        
                        # Recalculate coverage
                        total_activity_time = sum(item.get('duration', 0) for item in itinerary)
                        time_usage_percentage = (total_activity_time / duration) * 100
                        print(f"✅ Coverage after extension: {time_usage_percentage:.1f}% ({total_activity_time:.1f}/{duration:.1f} hours)")
                    else:
                        # If last activity is food or extension too large, raise error
                        print(f"📊 Debug - Location groups available:")
                        for loc_type, locs in location_groups.items():
                            print(f"  - {loc_type}: {len(locs)} locations")
                        
                        raise ValueError(
                            f"Unable to plan sufficient activities for {duration:.1f} hours. "
                            f"Only {total_activity_time:.1f} hours of activities available ({time_usage_percentage:.0f}% coverage). "
                            f"Try: 1) Selecting more interests, 2) Reducing exclusions, 3) Choosing a shorter duration (4-6 hours recommended), "
                            f"or 4) Selecting a different location with more nearby attractions."
                )
        
        # Calculate estimated cost
        estimated_cost = self._estimate_cost(itinerary, preferences.budget_tier)
        
        # Generate summary
        summary = self._generate_summary(itinerary, preferences)
        
        # Generate alternative suggestions
        used_locations = [item['location_obj'] for item in itinerary if 'location_obj' in item]
        alternatives = self._generate_alternatives(location_groups, preferences, used_locations)
        
        return DatePlan(
            itinerary=itinerary,
            total_duration=duration,
            estimated_cost=estimated_cost,
            summary=summary,
            alternative_suggestions=alternatives
        )
    
    def _group_locations_by_type(self, locations: List[Location]) -> Dict[str, List[Location]]:
        """Group locations by type for itinerary planning with smart filtering"""
        groups = {
            'food': [],
            'attraction': [],
            'activity': [],
            'heritage': []
        }
        
        for location in locations:
            if location.location_type in groups:
                groups[location.location_type].append(location)
        
        return groups
    
    def _is_appropriate_for_meal(self, location: Location, meal_type: str = None) -> bool:
        """Check if a location is appropriate for a meal (not an activity center)"""
        name_lower = location.name.lower()
        desc_lower = (location.description or '').lower()
        
        # Debug logging for problematic venues
        if 'jumbo' in name_lower or 'seafood' in name_lower:
            print(f"🐛 Checking: {location.name} for meal_type={meal_type}")
        
        # Keywords that indicate NOT a restaurant/cafe
        non_food_keywords = [
            'stadium', 'sports centre', 'sports center', 'gym', 'fitness',
            'playground', 'activity center', 'activity centre', 'amazonia',
            'museum', 'gallery', 'park', 'reserve', 'trail', 'zoo', 'wildlife'
        ]
        
        # Check if it's actually a food venue
        is_likely_food = any(keyword in name_lower or keyword in desc_lower for keyword in [
            'restaurant', 'cafe', 'coffee', 'dining', 'bistro', 'eatery', 
            'food court', 'hawker', 'kitchen', 'grill', 'bar', 'pub', 'kopitiam'
        ])
        
        # Not appropriate if it has non-food keywords and isn't clearly food
        has_non_food = any(keyword in name_lower for keyword in non_food_keywords)
        
        # Specific exclusions that are never appropriate for meals
        never_meal_keywords = [
            'kidzania', 'amazonia', 'wildlife', 'zoo', 'aquarium',
            'karaoke', 'ktv', 'k.t.v', 'karoke', 'karoke box'
        ]
        if any(keyword in name_lower for keyword in never_meal_keywords):
            return False
        
        # If it has non-food keywords but also has clear food indicators, allow it
        if has_non_food and not is_likely_food:
            return False
        
        # Special handling for breakfast and coffee breaks
        if meal_type in ['Coffee/Breakfast', 'Breakfast', 'Coffee Break']:
            # POSITIVE: What we WANT for breakfast/coffee
            breakfast_keywords = [
                'cafe', 'coffee', 'kopi', 'breakfast', 'brunch', 'bakery', 
                'toast', 'western', 'bistro', 'patisserie', 'sandwich', 'bagel',
                'espresso', 'latte', 'cappuccino', 'americano', 'tea house'
            ]
            has_breakfast_vibe = any(keyword in name_lower or keyword in desc_lower for keyword in breakfast_keywords)
            
            # NEGATIVE: What we DON'T WANT for breakfast/coffee
            non_breakfast_keywords = [
                'korean', 'chinese', 'indian', 'italian', 'french', 'japanese', 
                'thai', 'vietnamese', 'malay', 'seafood', 'steakhouse', 'steak',
                'fine dining', 'asian cuisine', 'peranakan', 'vegetarian restaurant',
                'noodles', 'ramen', 'sushi', 'dim sum', 'hotpot', 'bbq', 'grill'
            ]
            has_non_breakfast = any(keyword in name_lower or keyword in desc_lower for keyword in non_breakfast_keywords)
            
            # STRICT RULE: Must have breakfast vibe OR be hawker/food court
            if has_non_breakfast and not has_breakfast_vibe:
                # Only allow hawker centers and food courts (they usually have breakfast options)
                if any(keyword in name_lower for keyword in ['hawker', 'food court', 'kopitiam']):
                    if 'jumbo' in name_lower or 'seafood' in name_lower:
                        print(f"✅ {location.name}: Allowed (hawker/food court)")
                    return True
                else:
                    if 'jumbo' in name_lower or 'seafood' in name_lower:
                        print(f"❌ {location.name}: REJECTED for {meal_type} (has non-breakfast keywords, no breakfast vibe)")
                    return False  # Reject all other non-breakfast venues
            
            # If it has breakfast vibe, allow it
            if has_breakfast_vibe:
                return True
            
            # If no specific breakfast indicators but is a hawker/food court, allow
            if any(keyword in name_lower for keyword in ['hawker', 'food court', 'kopitiam']):
                return True
            
            # Otherwise, be conservative and reject
            return False
        
        # Special handling for lunch and dinner - EXCLUDE breakfast-only places and coffee shops
        if meal_type in ['Lunch', 'Dinner', 'Late Dinner']:
            # Breakfast-only places and coffee shops that should NOT serve lunch/dinner
            breakfast_only_keywords = [
                'ya kun', 'toast box', 'kaya toast', 'killiney', 'wang cafe',
                'kopitiam', 'yakun', 'toastbox', 'starbucks', 'coffee bean', 
                'the coffee bean', 'coffee bean & tea leaf', 'coffee bean and tea leaf',
                'starbuck', 'costa coffee', 'coffee shop'
            ]
            is_breakfast_only = any(keyword in name_lower for keyword in breakfast_only_keywords)
            
            if is_breakfast_only:
                if any(keyword in name_lower for keyword in ['ya kun', 'toast box', 'starbucks', 'coffee bean']):
                    print(f"❌ {location.name}: REJECTED for {meal_type} (breakfast/coffee-only establishment)")
                return False  # Don't serve lunch/dinner at breakfast places or coffee shops
            
        return True
    
    def _is_appropriate_for_date_type(self, location: Location, date_type: str) -> bool:
        """Check if location is appropriate for the date type"""
        name_lower = location.name.lower()
        desc_lower = (location.description or '').lower()
        
        # Romantic dates: HARD BLOCK zoo and river safari (not romantic at all!)
        if date_type == 'romantic':
            zoo_exclusions = ['zoo', 'river safari', 'river wonders', 'wildlife reserve', 'night safari']
            if any(keyword in name_lower or keyword in desc_lower for keyword in zoo_exclusions):
                return False
        
        # Romantic and cultural dates exclude child-focused venues
        if date_type in ['romantic', 'cultural']:
            child_exclusions = ['kids', 'children', 'junior', 'playground', 'family fun', 'family entertainment']
            if any(keyword in name_lower or keyword in desc_lower for keyword in child_exclusions):
                return False
        
        # Adventurous and casual dates can have any venue
        return True
    
    def _rank_by_date_type_match(self, locations: List[Location], date_type: str) -> List[Location]:
        """Re-rank food locations using simple keyword matching for date type"""
        # Date type keyword mapping (simple approach)
        date_type_keywords = {
            'romantic': ['romantic', 'intimate', 'fine dining', 'rooftop', 'waterfront', 'wine', 'candlelit', 'elegant', 'cozy', 'scenic'],
            'cultural': ['traditional', 'heritage', 'cultural', 'authentic', 'peranakan', 'local', 'historical', 'museum'],
            'adventurous': ['outdoor', 'adventure', 'unique', 'fusion', 'hawker', 'street food', 'market', 'experimental'],
            'casual': ['casual', 'relaxed', 'friendly', 'comfortable', 'bistro', 'cafe', 'laid-back', 'family']
        }
        
        keywords = date_type_keywords.get(date_type, [])
        if not keywords:
            return locations
        
        # Score each location by keyword matches
        scored = []
        for loc in locations:
            score = 0
            name_lower = loc.name.lower()
            desc_lower = (loc.description or '').lower()
            
            for keyword in keywords:
                if keyword in name_lower or keyword in desc_lower:
                    score += 1
            
            scored.append((score, loc))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Count matches
        matched = sum(1 for score, _ in scored if score > 0)
        print(f"  🎯 Date type keyword matching ({date_type}): {matched}/{len(locations)} venues matched")
        
        return [loc for _, loc in scored]
    
    def _create_time_based_itinerary(self, location_groups: Dict[str, List[Location]], preferences: UserPreferences, duration: float) -> List[Dict[str, Any]]:
        """Create itinerary based on actual time ranges and duration"""
        itinerary = []
        current_time = self._parse_time(preferences.start_time)
        if preferences.end_time:
            end_time = self._parse_time(preferences.end_time)
        else:
            end_time = self._add_hours(current_time, duration)
        
        # Sequential planning: plan each activity/meal after the previous one
        # Dynamic max activities based on duration (roughly 1.5 hours per activity/meal on average)
        max_activities = max(5, int(duration / 1.0) + 2)  # At least 5, scale with duration
        activity_count = 0
        
        print(f"📋 Planning itinerary for {duration:.1f} hours (max {max_activities} activities)")
        
        while self._time_difference(current_time, end_time) > 0.5 and activity_count < max_activities:  # At least 30 minutes remaining
            next_activity = self._plan_next_activity(location_groups, current_time, end_time, itinerary, preferences)
            if next_activity:
                # Check if this activity would exceed the end time (handles overnight dates correctly)
                time_left_after_activity = self._time_difference(next_activity['end_time'], end_time)
                
                if time_left_after_activity <= 0:
                    # Activity would go past end time, adjust it to end exactly at end_time
                    next_activity['end_time'] = end_time
                    next_activity['duration'] = self._time_difference(next_activity['start_time'], end_time)
                    itinerary.append(next_activity)
                    print(f"  🏁 Final activity adjusted to end at {end_time}")
                    break  # Stop planning after this activity
                else:
                    # Activity fits within time limit
                    itinerary.append(next_activity)
                    current_time = next_activity['end_time']
                    activity_count += 1
            else:
                print(f"⚠️ Could not plan next activity at {current_time} (ran out of locations)")
                break  # No more activities can be planned
        
        print(f"✅ Planned {len(itinerary)} activities totaling {sum(item.get('duration', 0) for item in itinerary):.1f} hours")
        return itinerary
    
    
    def _plan_next_activity(self, location_groups: Dict[str, List[Location]], current_time: str, end_time: str, existing_itinerary: List[Dict[str, Any]], preferences: UserPreferences) -> Optional[Dict[str, Any]]:
        """Plan the next activity/meal sequentially with proper travel time"""
        time_remaining = self._time_difference(current_time, end_time)
        if time_remaining < 0.5:  # Less than 30 minutes
            return None
        
        current_hour = int(current_time.split(':')[0])
        
        # PRIORITIZE MEALS: Always check if we should plan a meal first
        if self._should_plan_meal(current_time, existing_itinerary, preferences):
            meal_result = self._plan_next_meal(location_groups, current_time, time_remaining, existing_itinerary)
            if meal_result:  # If we can plan a meal, do it
                return meal_result
        
        # If no meal needed or couldn't plan meal, plan activity
        return self._plan_next_activity_only(location_groups, current_time, time_remaining, existing_itinerary, preferences)
    
    def _should_plan_meal(self, current_time: str, existing_itinerary: List[Dict[str, Any]], preferences: UserPreferences) -> bool:
        """Determine if we should plan a meal at this time"""
        current_hour = int(current_time.split(':')[0])
        
        # Check if user wants to exclude food activities
        if 'food' in self._current_exclusions:
            return False
        
        # Count existing meals and check for specific meal types
        meal_count = 0
        existing_meal_types = set()
        if existing_itinerary:
            for activity in existing_itinerary:
                if activity.get('type') == 'food':
                    meal_count += 1
                    existing_meal_types.add(activity.get('activity', ''))
        
        # Plan meals based on time windows and avoid duplicates
        if 6 <= current_hour <= 11 and 'Coffee/Breakfast' not in existing_meal_types:  # Breakfast/Coffee
            return True
        elif 12 <= current_hour <= 14 and 'Lunch' not in existing_meal_types:  # Lunch (12:00-14:00)
            return True
        elif 14 <= current_hour <= 16 and 'Coffee Break' not in existing_meal_types:  # Coffee Break (14:00-16:00)
            return True
        elif 17 <= current_hour <= 20 and 'Dinner' not in existing_meal_types:  # Dinner (17:00-20:00)
            return True
        elif current_hour >= 21 and 'Late Dinner' not in existing_meal_types:  # Late Dinner (after 21:00)
            return True
        
        return False
    
    def _plan_next_meal(self, location_groups: Dict[str, List[Location]], current_time: str, time_remaining: float, existing_itinerary: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Plan the next meal with travel time"""
        if not location_groups['food']:
            return None
        
        current_hour = int(current_time.split(':')[0])
        
        # Choose meal type and duration based on time and existing meals
        existing_meal_types = set()
        if existing_itinerary:
            for activity in existing_itinerary:
                if activity.get('type') == 'food':
                    existing_meal_types.add(activity.get('activity', ''))
        
        if 6 <= current_hour <= 11 and 'Coffee/Breakfast' not in existing_meal_types:
            meal_type = "Coffee/Breakfast"
            duration = 1.0
            food_index = 0
        elif 12 <= current_hour <= 14 and 'Lunch' not in existing_meal_types:
            meal_type = "Lunch"
            duration = 1.5  # Lunch duration (time window is 12:00-14:00)
            food_index = min(1, len(location_groups['food']) - 1)
        elif 14 <= current_hour <= 16 and 'Coffee Break' not in existing_meal_types:
            meal_type = "Coffee Break"
            duration = 1.0
            food_index = min(2, len(location_groups['food']) - 1)
        elif 17 <= current_hour <= 20 and 'Dinner' not in existing_meal_types:
            meal_type = "Dinner"
            duration = 2.0
            food_index = min(3, len(location_groups['food']) - 1)
        elif current_hour >= 21 and 'Late Dinner' not in existing_meal_types:
            meal_type = "Late Dinner"
            duration = 2.0
            food_index = 0
        else:
            return None  # Don't plan duplicate meal types
        
        # Get food location (avoid duplicates, prefer cafes for coffee breaks)
        used_location_ids = set()
        for a in existing_itinerary:
            if isinstance(a, dict) and 'location_obj' in a:
                used_location_ids.add(a['location_obj'].id)
            elif hasattr(a, 'id'):  # Direct Location object
                used_location_ids.add(a.id)
        
        available_food = [loc for loc in location_groups['food'] if loc.id not in used_location_ids]
        
        if not available_food:
            # If all food locations used, allow reuse but prefer different ones
            available_food = location_groups['food']
        
        # Filter locations based on meal type appropriateness
        meal_appropriate_food = [loc for loc in available_food if self._is_appropriate_for_meal(loc, meal_type)]
        
        print(f"🍽️ Planning {meal_type}: {len(available_food)} total, {len(meal_appropriate_food)} appropriate")
        
        # CRITICAL: Never fall back to inappropriate venues for breakfast/coffee
        if not meal_appropriate_food:
            if meal_type in ['Coffee/Breakfast', 'Breakfast', 'Coffee Break']:
                # For breakfast/coffee, only allow hawker centers as last resort
                hawker_fallback = [loc for loc in available_food 
                                  if any(keyword in loc.name.lower() for keyword in ['hawker', 'food court', 'kopitiam'])]
                if hawker_fallback:
                    print(f"⚠️ No breakfast cafes found, using hawker centers as fallback")
                    meal_appropriate_food = hawker_fallback
                else:
                    print(f"❌ ERROR: No breakfast-appropriate venues found!")
                    return None  # Don't plan inappropriate meals
            else:
                # For lunch/dinner, fallback is okay
                meal_appropriate_food = available_food
        
        if meal_type == "Coffee/Breakfast":
            # Prioritize breakfast/cafe places - check both name AND description
            breakfast_keywords = [
                'cafe', 'coffee', 'kopi', 'kopitiam', 'bistro', 'brunch', 
                'breakfast', 'bakery', 'toast', 'patisserie', 'espresso', 'latte'
            ]
            breakfast_locations = [loc for loc in meal_appropriate_food 
                                 if any(keyword in loc.name.lower() or keyword in (loc.description or '').lower() 
                                       for keyword in breakfast_keywords)]
            
            # Filter out cafes that typically don't open early (before 10 AM)
            late_opening_keywords = ['hoshino', 'specialty coffee', 'artisan coffee', 'third wave']
            early_breakfast_locations = [
                loc for loc in breakfast_locations 
                if not any(keyword in loc.name.lower() or keyword in (loc.description or '').lower() 
                          for keyword in late_opening_keywords)
            ]
            
            print(f"☕ Found {len(breakfast_locations)} breakfast locations, {len(early_breakfast_locations)} open early")
            
            if early_breakfast_locations:
                print(f"✅ Selected breakfast: {early_breakfast_locations[0].name}")
                food_location = early_breakfast_locations[0]  # Take early-opening cafe
            elif breakfast_locations:
                # If only late cafes available, use them anyway
                print(f"⚠️ Using late-opening cafe: {breakfast_locations[0].name}")
                food_location = breakfast_locations[0]
            else:
                # Fallback to hawker centers
                hawker_locations = [loc for loc in meal_appropriate_food
                                  if any(keyword in loc.name.lower() for keyword in ['hawker', 'food court'])]
                if hawker_locations:
                    food_location = hawker_locations[0]
                else:
                    food_location = meal_appropriate_food[min(food_index, len(meal_appropriate_food) - 1)]
        elif meal_type == "Coffee Break":
            # STRICT: Only cafes and coffee shops - check both name AND description
            cafe_keywords = [
                'cafe', 'coffee', 'kopi', 'kopitiam', 'bistro', 'espresso', 
                'latte', 'cappuccino', 'americano', 'tea house', 'patisserie'
            ]
            cafe_locations = [loc for loc in meal_appropriate_food 
                            if any(keyword in loc.name.lower() or keyword in (loc.description or '').lower() 
                                  for keyword in cafe_keywords)]
            if cafe_locations:
                food_location = cafe_locations[0]  # Take the first/best cafe
            else:
                # If no cafes found, prefer hawker centers over restaurants
                hawker_locations = [loc for loc in meal_appropriate_food
                                  if any(keyword in loc.name.lower() for keyword in ['hawker', 'food court', 'kopitiam'])]
                if hawker_locations:
                    food_location = hawker_locations[0]
        else:
                    food_location = meal_appropriate_food[min(food_index, len(meal_appropriate_food) - 1)]
        
        # Calculate travel time FIRST (before adjusting duration)
        start_time = current_time
        travel_time = 0.0
        if existing_itinerary:
            last_location = existing_itinerary[-1].get('location_obj')
            if last_location:
                travel_time = self._calculate_travel_time(last_location, food_location)
                start_time = self._add_hours(current_time, travel_time)
        
        # Adjust meal duration to account for travel time AND remaining time
        # Available time = time_remaining - travel_time
        available_time = time_remaining - travel_time
        if duration > available_time:
            duration = max(0.5, available_time)  # Minimum 30 minutes for any meal
            print(f"  ⏱️ Adjusted {meal_type} duration to {duration:.1f}h (travel: {travel_time:.1f}h, available: {available_time:.1f}h)")
        
        end_time = self._add_hours(start_time, duration)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'activity': meal_type,
            'location': food_location.name,
            'address': food_location.address or 'Address not available',
            'type': 'food',
            'duration': duration,
            'description': f"{food_location.description[:100]}...",
            'location_obj': food_location
        }
    
    def _plan_next_activity_only(self, location_groups: Dict[str, List[Location]], current_time: str, time_remaining: float, existing_itinerary: List[Dict[str, Any]], preferences: UserPreferences = None) -> Optional[Dict[str, Any]]:
        """Plan the next non-meal activity with travel time, considering date type preferences"""
        # Count existing activities to avoid too many of the same type
        activity_count = len([a for a in existing_itinerary if a.get('type') != 'food'])
        
        # Count sports activities specifically (NOT including walks)
        sports_count = len([a for a in existing_itinerary 
                           if a.get('type') == 'activity' or 
                           (a.get('activity') and 'sports' in a.get('activity', '').lower() and 
                            a.get('activity') != 'Walk')])
        
        # Check for user exclusions (direct from exclusions array)
        exclude_sports = 'sports' in self._current_exclusions
        exclude_cultural = 'cultural' in self._current_exclusions
        exclude_nature = 'nature' in self._current_exclusions
        
        # Get date type for activity prioritization
        date_type = preferences.date_type if preferences else 'casual'
        
        # Get used location IDs to avoid duplicates
        used_location_ids = set()
        for a in existing_itinerary:
            if isinstance(a, dict) and 'location_obj' in a:
                used_location_ids.add(a['location_obj'].id)
            elif hasattr(a, 'id'):  # Direct Location object
                used_location_ids.add(a.id)
        
        # Find available locations (not used before)
        available_activities = [loc for loc in location_groups.get('activity', []) if loc.id not in used_location_ids]
        available_attractions = [loc for loc in location_groups.get('attraction', []) if loc.id not in used_location_ids]
        available_heritage = [loc for loc in location_groups.get('heritage', []) if loc.id not in used_location_ids]
        
        # Apply date type filter to attractions (e.g., exclude zoo/river safari for romantic dates)
        available_attractions = [loc for loc in available_attractions if self._is_appropriate_for_date_type(loc, date_type)]
        
        # DEDUPLICATION: If zoo or river safari already used, exclude the other
        zoo_safari_keywords = ['singapore zoo', 'river safari', 'river wonders', 'night safari', 'wildlife reserves singapore']
        has_zoo_or_safari = any(
            any(keyword in a.get('location', '').lower() for keyword in zoo_safari_keywords)
            for a in existing_itinerary if isinstance(a, dict)
        )
        
        if has_zoo_or_safari:
            # Filter out all zoo/safari attractions
            available_attractions = [
                loc for loc in available_attractions 
                if not any(keyword in loc.name.lower() for keyword in zoo_safari_keywords)
            ]
            print(f"  🚫 Zoo/Safari deduplication: Excluded similar attractions (one already planned)")
        
        # For very long dates (12+ hours), allow location reuse if we run out of new locations
        duration = preferences.get_duration_hours()
        if duration >= 12.0 and not available_activities and not available_attractions and not available_heritage:
            print(f"  🔄 Long date ({duration:.1f}h): Allowing location reuse for activities")
            # Allow reuse - just use all locations again (but still apply date type filter)
            available_activities = location_groups.get('activity', [])
            available_attractions = [loc for loc in location_groups.get('attraction', []) if self._is_appropriate_for_date_type(loc, date_type)]
            available_heritage = location_groups.get('heritage', [])
        
        # Choose activity type based on date type preferences, time, and what's available
        location = None
        activity_duration = 1.0
        activity_type = None
        
        # DATE TYPE PRIORITIZATION:
        # Adventurous → prioritize sports/activities and nature walks
        # Cultural → prioritize museums/heritage sites and cultural attractions
        # Romantic/Casual → use normal flow
        
        # Date-type specific prioritization using date_vibe
        if date_type == 'adventurous':
            # ADVENTUROUS: Prioritize sports/activities first (max 1), then adventurous-vibe attractions
            if location is None and time_remaining >= 2.0 and available_activities and not exclude_sports and sports_count < 1:
                # Activities don't have date_vibe (fallback), so select first available
                location = available_activities[0]
                activity_duration = min(2.0, time_remaining)
                activity_type = self._get_simple_activity_type(location)
                print(f"  🏃 Adventurous date: Selected sports activity (1/{1} max)")
            elif location is None and time_remaining >= 1.5 and available_attractions and not exclude_nature:
                # Prioritize attractions with 'adventurous' date_vibe, then fallback to no-vibe locations
                adventurous_attractions = [loc for loc in available_attractions 
                                          if loc.date_vibe and 'adventurous' in loc.date_vibe]
                fallback_attractions = [loc for loc in available_attractions 
                                       if not loc.date_vibe or len(loc.date_vibe) == 0]
                
                # Try adventurous-vibe attractions first
                for attraction in (adventurous_attractions + fallback_attractions):
                    activity_type = self._get_attraction_activity_type(attraction, exclude_nature, exclude_cultural)
                    if activity_type == 'Walk':  # Prefer walks for adventurous
                        location = attraction
                        activity_duration = min(2.0, time_remaining)
                        vibe_match = "vibe-matched" if attraction.date_vibe and 'adventurous' in attraction.date_vibe else "fallback"
                        print(f"  🚶 Adventurous date: Selected nature walk ({vibe_match})")
                        break
        
        elif date_type == 'cultural':
            # CULTURAL: Prioritize heritage sites and cultural-vibe attractions
            if location is None and time_remaining >= 1.5 and available_heritage and not exclude_cultural:
                # Heritage sites don't have date_vibe (fallback), so select first available
                location = available_heritage[0]
                activity_duration = min(1.5, time_remaining)
                activity_type = 'Heritage Walk'
                print(f"  🏛️ Cultural date: Selected heritage site (fallback)")
            elif location is None and time_remaining >= 1.5 and available_attractions and not exclude_cultural:
                # Prioritize attractions with 'cultural' date_vibe, then fallback to no-vibe locations
                cultural_attractions = [loc for loc in available_attractions 
                                       if loc.date_vibe and 'cultural' in loc.date_vibe]
                fallback_attractions = [loc for loc in available_attractions 
                                       if not loc.date_vibe or len(loc.date_vibe) == 0]
                
                # Try cultural-vibe attractions first
                for attraction in (cultural_attractions + fallback_attractions):
                    activity_type = self._get_attraction_activity_type(attraction, exclude_nature, exclude_cultural)
                    if activity_type == 'Cultural Visit':  # Prefer cultural visits
                        location = attraction
                        activity_duration = min(2.0, time_remaining)
                        vibe_match = "vibe-matched" if attraction.date_vibe and 'cultural' in attraction.date_vibe else "fallback"
                        print(f"  🎨 Cultural date: Selected cultural attraction ({vibe_match})")
                        break
        
        elif date_type == 'romantic':
            # ROMANTIC: Prioritize romantic-vibe attractions over generic locations
            if location is None and time_remaining >= 1.5 and available_attractions:
                # Prioritize attractions with 'romantic' date_vibe, then fallback to no-vibe locations
                romantic_attractions = [loc for loc in available_attractions 
                                       if loc.date_vibe and 'romantic' in loc.date_vibe]
                fallback_attractions = [loc for loc in available_attractions 
                                       if not loc.date_vibe or len(loc.date_vibe) == 0]
                
                # Try romantic-vibe attractions first (prefer non-Walk activities for romantic dates)
                for attraction in (romantic_attractions + fallback_attractions):
                    activity_type = self._get_attraction_activity_type(attraction, exclude_nature, exclude_cultural)
                    if activity_type in ['Cultural Visit', 'Visit']:  # Prefer indoor/cultural for romantic
                        location = attraction
                        activity_duration = min(2.0, time_remaining)
                        vibe_match = "vibe-matched" if attraction.date_vibe and 'romantic' in attraction.date_vibe else "fallback"
                        print(f"  💕 Romantic date: Selected romantic attraction ({vibe_match})")
                        break
        
        elif date_type == 'casual':
            # CASUAL: Prioritize casual-vibe attractions (most flexible)
            if location is None and time_remaining >= 1.5 and available_attractions:
                # Prioritize attractions with 'casual' date_vibe, then fallback to no-vibe locations
                casual_attractions = [loc for loc in available_attractions 
                                     if loc.date_vibe and 'casual' in loc.date_vibe]
                fallback_attractions = [loc for loc in available_attractions 
                                       if not loc.date_vibe or len(loc.date_vibe) == 0]
                
                # Try casual-vibe attractions first (any activity type is fine for casual)
                combined_attractions = casual_attractions + fallback_attractions
                if combined_attractions:
                    location = combined_attractions[0]
                    activity_duration = min(2.0, time_remaining)
                    activity_type = self._get_attraction_activity_type(location, exclude_nature, exclude_cultural)
                    vibe_match = "vibe-matched" if location.date_vibe and 'casual' in location.date_vibe else "fallback"
                    print(f"  😊 Casual date: Selected casual attraction ({vibe_match})")
        
        # FALLBACK: Standard selection for romantic/casual OR if date-type-specific search failed
        if location is None and time_remaining >= 2.0 and available_attractions:
            # Check each attraction to find one that's not excluded
            for attraction in available_attractions:
                activity_type = self._get_attraction_activity_type(attraction, exclude_nature, exclude_cultural)
                if activity_type is not None:  # Found a valid attraction
                    location = attraction
                    activity_duration = min(2.0, time_remaining)
                    break
        
        if location is None and time_remaining >= 2.0 and available_activities and sports_count < 1 and not exclude_sports:
            location = available_activities[0]
            activity_duration = min(2.0, time_remaining)
            activity_type = self._get_simple_activity_type(location)
        elif location is None and time_remaining >= 1.5 and available_heritage and not exclude_cultural:
            location = available_heritage[0]
            activity_duration = min(1.5, time_remaining)
            activity_type = 'Heritage Walk'
        elif location is None and time_remaining >= 1.0 and available_attractions:
            # Check each attraction to find one that's not excluded
            for attraction in available_attractions:
                activity_type = self._get_attraction_activity_type(attraction, exclude_nature, exclude_cultural)
                if activity_type is not None:  # Found a valid attraction
                    location = attraction
                    activity_duration = min(1.5, time_remaining)
                    break
        
        if location is None and time_remaining >= 1.0 and available_activities and sports_count < 1 and not exclude_sports:
            location = available_activities[0]
            activity_duration = min(1.5, time_remaining)
            activity_type = self._get_simple_activity_type(location)
        
        if location is None:
            return None
        
        # Calculate travel time FIRST (before finalizing duration)
        start_time = current_time
        travel_time = 0.0
        if existing_itinerary:
            last_location = existing_itinerary[-1].get('location_obj')
            if last_location:
                travel_time = self._calculate_travel_time(last_location, location)
                start_time = self._add_hours(current_time, travel_time)
        
        # Adjust activity duration to account for travel time AND remaining time
        # Available time = time_remaining - travel_time
        available_time = time_remaining - travel_time
        if activity_duration > available_time:
            activity_duration = max(0.5, available_time)  # Minimum 30 minutes for any activity
            print(f"  ⏱️ Adjusted activity duration to {activity_duration:.1f}h (travel: {travel_time:.1f}h, available: {available_time:.1f}h)")
        
        end_time = self._add_hours(start_time, activity_duration)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'activity': activity_type,
            'location': location.name,
            'address': location.address or 'Address not available',
            'type': location.location_type,
            'duration': activity_duration,
            'description': f"{location.description[:100]}...",
            'location_obj': location
        }
    
    def _plan_meals_by_time(self, location_groups: Dict[str, List[Location]], start_time: str, duration: float) -> List[Dict[str, Any]]:
        """Plan meals based on actual time ranges"""
        meals = []
        current_time = start_time
        
        if not location_groups['food']:
            return meals
        
        # Parse start time to get hour
        start_hour = int(start_time.split(':')[0])
        
        # Breakfast/Coffee: 6:00 - 11:00
        if 6 <= start_hour <= 11:
            if location_groups['food']:
                food_location = location_groups['food'][0]  # Use first available food location
                end_time = self._add_hours(current_time, 1.0)
                meals.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'activity': 'Coffee/Breakfast',
                    'location': food_location.name,
                    'address': food_location.address or 'Address not available',
                    'type': 'food',
                    'duration': 1.0,
                    'description': f"Start your day with {food_location.description[:100]}...",
                    'location_obj': food_location
                })
                current_time = end_time
        
        # Lunch: 12:00 - 14:00 (if date spans this time)
        date_end_time = self._add_hours(start_time, duration)
        if start_hour <= 12 and self._time_after_or_equal(date_end_time, "12:00"):  # Date spans lunch time
            if len(location_groups['food']) > 1:
                lunch_location = location_groups['food'][1]  # Use second food location to avoid duplicate
                # Add travel time
                if meals:
                    travel_time = self._calculate_travel_time(meals[-1]['location_obj'], lunch_location)
                    current_time = self._add_hours(current_time, travel_time)
                
                end_time = self._add_hours(current_time, 1.5)
                meals.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'activity': 'Lunch',
                    'location': lunch_location.name,
                    'address': lunch_location.address or 'Address not available',
                    'type': 'food',
                    'duration': 1.5,
                    'description': f"{lunch_location.description[:100]}...",
                    'location_obj': lunch_location
                })
                current_time = end_time
        
        # Coffee Break: 14:00 - 17:00 (for extended dates)
        if start_hour <= 14 and duration > 6:  # Date starts before 2pm and is long enough
            if len(location_groups['food']) > 2:
                coffee_location = location_groups['food'][2]  # Use third food location to avoid duplicate
                # Add travel time
                if meals:
                    travel_time = self._calculate_travel_time(meals[-1]['location_obj'], coffee_location)
                    current_time = self._add_hours(current_time, travel_time)
                
                end_time = self._add_hours(current_time, 1.0)
                meals.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'activity': 'Coffee Break',
                    'location': coffee_location.name,
                    'address': coffee_location.address or 'Address not available',
                    'type': 'food',
                    'duration': 1.0,
                    'description': f"Relax with coffee at {coffee_location.description[:100]}...",
                    'location_obj': coffee_location
                })
                current_time = end_time
        
        # Dinner: 17:00 - 19:30 (if date spans this time)
        if (start_hour <= 17 and self._time_after_or_equal(date_end_time, "17:01")) or (start_hour >= 17 and self._time_after_or_equal(date_end_time, "17:00")):  # Date spans dinner time
            # Use different food location to avoid duplicates
            dinner_index = 3 if len(location_groups['food']) > 3 else min(2, len(location_groups['food']) - 1)
            if len(location_groups['food']) > dinner_index:
                dinner_location = location_groups['food'][dinner_index]
                
                # Plan dinner for the evening time slot (19:00-21:00 for 7-hour dates)
                dinner_start_time = "19:00" if duration >= 7 else self._add_hours(start_time, duration - 2.0)
                
                # Add travel time if there are previous meals
                if meals:
                    travel_time = self._calculate_travel_time(meals[-1]['location_obj'], dinner_location)
                    dinner_start_time = self._add_hours(dinner_start_time, travel_time)
                
                dinner_end_time = self._add_hours(dinner_start_time, 2.0)
                
                meals.append({
                    'start_time': dinner_start_time,
                    'end_time': dinner_end_time,
                    'activity': 'Dinner',
                    'location': dinner_location.name,
                    'address': dinner_location.address or 'Address not available',
                    'type': 'food',
                    'duration': 2.0,
                    'description': f"{dinner_location.description[:100]}...",
                    'location_obj': dinner_location
                })
        
        # Late Dinner: 21:00 - 02:00 (if date spans this time)
        if start_hour >= 21 or (start_hour <= 2 and duration > 2):  # Date starts after 9pm or before 2am
            late_dinner_index = 0
            if len(location_groups['food']) > late_dinner_index:
                late_dinner_location = location_groups['food'][late_dinner_index]
                # Add travel time
                if meals:
                    travel_time = self._calculate_travel_time(meals[-1]['location_obj'], late_dinner_location)
                    current_time = self._add_hours(current_time, travel_time)
                
                end_time = self._add_hours(current_time, 2.0)
                meals.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'activity': 'Late Dinner',
                    'location': late_dinner_location.name,
                    'address': late_dinner_location.address or 'Address not available',
                    'type': 'food',
                    'duration': 2.0,
                    'description': f"Late night dining at {late_dinner_location.description[:100]}...",
                    'location_obj': late_dinner_location
                })
                current_time = end_time
        
        return meals
    
    def _plan_activities_by_time(self, location_groups: Dict[str, List[Location]], start_time: str, duration: float, meals_planned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan activities to fill remaining time between meals"""
        activities = []
        current_time = start_time
        
        if not meals_planned:
            # No meals planned, fill entire duration with activities
            # Plan multiple activities to fill the duration
            remaining_duration = duration
            activity_count = 0
            
            while remaining_duration > 0.5 and activity_count < 3:  # Max 3 activities
                # Choose activity type based on remaining time and what's available
                if remaining_duration >= 2.0 and location_groups['attraction'] and activity_count < 1:
                    # Long activity - use attraction (max 1)
                    activity_duration = min(2.0, remaining_duration)
                    location = location_groups['attraction'][activity_count]  # Use different attraction each time
                    activity_type = 'Walk' if 'walk' in location.name.lower() or 'park' in location.name.lower() else 'Cultural Visit'
                elif remaining_duration >= 1.0 and location_groups['activity'] and activity_count < 1:
                    # Medium activity - use sports/activity (max 1)
                    activity_duration = min(2.0, remaining_duration)
                    location = location_groups['activity'][activity_count]  # Use different activity each time
                    activity_type = self._get_simple_activity_type(location)
                elif location_groups['attraction'] and activity_count < 2:
                    # Short activity - use attraction
                    activity_duration = min(1.0, remaining_duration)
                    location = location_groups['attraction'][activity_count]  # Use different attraction each time
                    activity_type = 'Walk' if 'walk' in location.name.lower() or 'park' in location.name.lower() else 'Cultural Visit'
                else:
                    break
                
                # Add travel time if there are previous activities
                if activities:
                    travel_time = self._calculate_travel_time(activities[-1]['location_obj'], location)
                    current_time = self._add_hours(current_time, travel_time)
                
                end_time = self._add_hours(current_time, activity_duration)
                activities.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'activity': activity_type,
                    'location': location.name,
                    'address': location.address or 'Address not available',
                    'type': location.location_type,
                    'duration': activity_duration,
                    'description': f"{location.description[:100]}...",
                    'location_obj': location
                })
                
                current_time = end_time
                remaining_duration -= activity_duration
                activity_count += 1
        else:
            # Fill gaps between meals with activities
            for i, meal in enumerate(meals_planned):
                # Add activity before meal if there's time
                time_until_meal = self._time_difference(current_time, meal['start_time'])
                if time_until_meal > 0.5:  # At least 30 minutes
                    activity = self._create_activity_for_duration(location_groups, current_time, time_until_meal)
                    if activity:
                        activities.append(activity)
                        current_time = activity['end_time']
                
                # Move to after meal
                current_time = meal['end_time']
            
            # Add final activity if there's remaining time
            # Calculate the actual end time of the date
            actual_end_time = self._add_hours(start_time, duration)
            time_remaining = self._time_difference(current_time, actual_end_time)
            if time_remaining > 0.5:
                # Add travel time if there are previous meals/activities
                if meals_planned or activities:
                    # Get the last location (either from last meal or last activity)
                    last_location = None
                    if activities:
                        last_location = activities[-1]['location_obj']
                    elif meals_planned:
                        last_location = meals_planned[-1]['location_obj']
                    
                    if last_location:
                        # Find the next activity location
                        if location_groups['activity']:
                            next_location = location_groups['activity'][0]
                            travel_time = self._calculate_travel_time(last_location, next_location)
                            current_time = self._add_hours(current_time, travel_time)
                
                final_activity = self._create_activity_for_duration(location_groups, current_time, time_remaining)
                if final_activity:
                    activities.append(final_activity)
        
        return activities
    
    def _create_activity_for_duration(self, location_groups: Dict[str, List[Location]], start_time: str, duration: float) -> Optional[Dict[str, Any]]:
        """Create an appropriate activity for the given duration"""
        if duration < 0.5:  # Less than 30 minutes
            return None
        
        # Prefer attractions for shorter durations, activities for longer ones
        if duration <= 2.0 and location_groups['attraction']:
            location = location_groups['attraction'][0]
            activity_type = 'Walk' if 'walk' in location.name.lower() or 'park' in location.name.lower() else 'Cultural Visit'
        elif duration > 2.0 and location_groups['activity']:
            location = location_groups['activity'][0]
            activity_type = self._get_simple_activity_type(location)
        elif location_groups['attraction']:
            location = location_groups['attraction'][0]
            activity_type = 'Walk' if 'walk' in location.name.lower() or 'park' in location.name.lower() else 'Cultural Visit'
        else:
            return None
        
        end_time = self._add_hours(start_time, duration)
        return {
            'start_time': start_time,
            'end_time': end_time,
            'activity': activity_type,
            'location': location.name,
            'address': location.address or 'Address not available',
            'type': location.location_type,
            'duration': duration,
            'description': f"{location.description[:100]}...",
            'location_obj': location
        }
    
    def _time_difference(self, start_time: str, end_time: str) -> float:
        """
        Calculate time difference in hours between two time strings.
        Returns positive if end_time is after start_time, negative if before.
        
        For overnight dates, only treats as next day if difference would be > 12 hours backward.
        This prevents false positives like 20:00 to 18:00 being treated as overnight.
        """
        start_hour, start_min = map(int, start_time.split(':'))
        end_hour, end_min = map(int, end_time.split(':'))
        
        start_total_min = start_hour * 60 + start_min
        end_total_min = end_hour * 60 + end_min
        
        # Calculate same-day difference first
        same_day_diff = end_total_min - start_total_min
        
        # Only treat as overnight if:
        # 1. End time is before start time (negative difference)
        # 2. The backward difference is >= 12 hours (indicates genuine overnight)
        # This prevents 20:00 -> 18:00 (-2h) from being treated as overnight (+22h)
        if same_day_diff < 0 and abs(same_day_diff) >= 12 * 60:
            # Genuine overnight: add 24 hours to end time
            end_total_min += 24 * 60
            return (end_total_min - start_total_min) / 60.0
        else:
            # Same day or small backward difference: return as-is (can be negative)
            return same_day_diff / 60.0
    
    def _time_after_or_equal(self, time1: str, time2: str) -> bool:
        """Check if time1 is after or equal to time2 (same day)"""
        hour1, min1 = map(int, time1.split(':'))
        hour2, min2 = map(int, time2.split(':'))
        
        total_min1 = hour1 * 60 + min1
        total_min2 = hour2 * 60 + min2
        
        return total_min1 >= total_min2
    
        return itinerary
    
    def _parse_time(self, time_str: str) -> str:
        """Parse time string (basic implementation)"""
        return time_str
    
    def _add_hours(self, time_str: str, hours: float) -> str:
        """Add hours to time string (basic implementation)"""
        try:
            hour, minute = map(int, time_str.split(':'))
            total_minutes = hour * 60 + minute + int(hours * 60)
            new_hour = (total_minutes // 60) % 24
            new_minute = total_minutes % 60
            return f"{new_hour:02d}:{new_minute:02d}"
        except:
            return time_str
    
    def _calculate_travel_time(self, location1: Location, location2: Location) -> float:
        """Calculate travel time between two locations in hours"""
        if not location1.coordinates or not location2.coordinates:
            return 0.25  # Default 15 minutes if coordinates missing
        
        # Extract coordinates (longitude, latitude)
        lon1, lat1 = location1.coordinates
        lon2, lat2 = location2.coordinates
        
        # Calculate distance using Haversine formula
        import math
        
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = 6371 * c  # Earth's radius in km
        
        # Estimate travel time based on distance
        # Assume average speed of 30 km/h in Singapore (including traffic, public transport)
        travel_time_hours = distance_km / 30.0
        
        # Add minimum travel time and cap maximum
        travel_time_hours = max(0.1, min(travel_time_hours, 1.0))  # 6 minutes to 1 hour
        
        return round(travel_time_hours, 2)
    
    def _create_activity_dict(self, location: Location, current_time: str, duration: float, activity_type: str, existing_itinerary: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create activity dictionary with travel time"""
        # Add travel time if there are previous activities
        start_time = current_time
        if existing_itinerary:
            last_location = existing_itinerary[-1].get('location_obj')
            if last_location:
                travel_time = self._calculate_travel_time(last_location, location)
                start_time = self._add_hours(current_time, travel_time)
        
        end_time = self._add_hours(start_time, duration)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'activity': activity_type,
            'location': location.name,
            'address': location.address or 'Address not available',
            'type': location.location_type,
            'duration': duration,
            'description': f"{location.description[:100]}...",
            'location_obj': location
        }
    
    def _get_simple_activity_type(self, location: Location) -> str:
        """Simple activity type determination"""
        if location.location_type == 'activity':
            return 'Sports Activity'
        elif location.location_type == 'attraction':
            return 'Cultural Visit'
        elif location.location_type == 'heritage':
            return 'Heritage Site'
        else:
            return 'Activity'
    
    def _get_attraction_activity_type(self, location: Location, exclude_nature: bool = False, exclude_cultural: bool = False) -> Optional[str]:
        """Determine activity type for attraction based on its characteristics, returns None if should be excluded"""
        name_lower = location.name.lower()
        desc_lower = (location.description or '').lower()
        text_to_search = f"{name_lower} {desc_lower}"
        
        # Check if it's a nature location
        nature_keywords = ['walk', 'park', 'nature', 'reserve', 'garden', 'botanical', 'trail', 'outdoor']
        if any(keyword in text_to_search for keyword in nature_keywords):
            if exclude_nature:
                return None  # Skip nature locations if excluded
            return 'Walk'
        
        # Check if it's explicitly cultural
        cultural_keywords = ['museum', 'gallery', 'art', 'heritage', 'historical', 'cultural', 'exhibition', 'temple', 'worship', 'church', 'mosque', 'shrine', 'cathedral']
        if any(keyword in text_to_search for keyword in cultural_keywords):
            if exclude_cultural:
                return None  # Skip cultural locations if excluded
            return 'Cultural Visit'
        
        # Check if it's a shopping location
        shopping_keywords = ['shopping', 'shop', 'mall', 'retail', 'boutique', 'orchard road', 'store']
        if any(keyword in text_to_search for keyword in shopping_keywords):
            return 'Shopping'  # ✅ Always allowed (not in exclusions)
        
        # Default: general attraction (always allowed)
        return 'Attraction Visit'
    
    def _estimate_cost(self, itinerary: List[Dict[str, Any]], budget_tier: str) -> str:
        """Estimate total cost based on budget tier and meal-specific pricing"""
        total_min = 0
        total_max = 0
        
        # Meal-specific costs per person
        meal_costs = {
            "Coffee/Breakfast": (10, 10),     # Fixed $10 per person
            "Coffee Break": (10, 10),         # Fixed $10 per person
            "Lunch": (self._get_budget_cost(budget_tier, 0.8)),  # 80% of budget tier
            "Dinner": (self._get_budget_cost(budget_tier, 1.0)),  # Full budget tier
            "Late Dinner": (self._get_budget_cost(budget_tier, 1.0)),  # Full budget tier
        }
        
        for item in itinerary:
            if item.get('type') == 'food':
                meal_type = item.get('activity', '')
                if meal_type in meal_costs:
                    cost_range = meal_costs[meal_type]
                    total_min += cost_range[0]
                    total_max += cost_range[1]
        
        if total_min == 0 and total_max == 0:
            return "Minimal cost (no meals planned)"
        
        return f"${total_min}-${total_max} per person"
    
    def _get_budget_cost(self, budget_tier: str, multiplier: float = 1.0) -> tuple:
        """Get cost range for a budget tier with multiplier"""
        base_costs = {
            "$": (10, 15),
            "$$": (20, 40),
            "$$$": (50, 70)
        }
        
        base = base_costs.get(budget_tier, (20, 40))
        return (int(base[0] * multiplier), int(base[1] * multiplier))
    
    def _generate_summary(self, itinerary: List[Dict[str, Any]], preferences: UserPreferences) -> str:
        """Generate a summary of the date plan"""
        summary = f"Your {preferences.get_duration_hours():.1f}-hour {preferences.date_type} date:\n\n"
        
        for activity in itinerary:
            start_time = activity.get('start_time', activity.get('time', 'Unknown'))
            summary += f"• {start_time}: {activity['activity']} at {activity['location']}\n"
        
        # Custom taglines for each date type
        taglines = {
            'romantic': "An intimate journey designed to create lasting memories together.",
            'casual': "A relaxed and enjoyable day exploring what Singapore has to offer.",
            'adventurous': "An exciting experience filled with unique discoveries and flavors.",
            'cultural': "A curated exploration of tradition, heritage, and authenticity."
        }
        
        tagline = taglines.get(preferences.date_type, "A thoughtfully planned experience curated just for you.")
        summary += f"\n{tagline}"
        
        return summary
    
    def _generate_alternatives(self, location_groups: Dict[str, List[Location]], preferences: UserPreferences, used_locations: List[Location]) -> List[str]:
        """Generate alternative suggestions that don't repeat itinerary locations"""
        alternatives = []
        used_location_ids = {loc.id for loc in used_locations}
        
        for location_type, locations in location_groups.items():
            # Find locations of this type that weren't used in the itinerary
            unused_locations = [loc for loc in locations if loc.id not in used_location_ids]
            
            if unused_locations:
                # Take the first unused location of this type
                alt_location = unused_locations[0]
                address = alt_location.address or "Address not available"
                alternatives.append(f"Alternative {location_type}: {alt_location.name} - {address}")
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def get_processing_summary(self, result: DatePlanResult) -> str:
        """Get a comprehensive summary of the entire planning process"""
        summary = "🎯 AI Date Planning Summary\n"
        summary += "=" * 50 + "\n\n"
        
        # Processing stats
        stats = result.processing_stats
        summary += f"📊 Processing Statistics:\n"
        summary += f"  • Total locations loaded: {stats['total_locations']}\n"
        summary += f"  • After rule filtering: {stats['filtered_locations']}\n"
        summary += f"  • After AI relevance: {stats['relevant_locations']}\n"
        summary += f"  • Final activities: {stats['final_activities']}\n"
        summary += f"  • Embeddings ready: {stats['embeddings_ready']}\n\n"
        
        # Date plan
        plan = result.date_plan
        summary += f"📅 Your Date Plan:\n"
        summary += f"  • Duration: {plan.total_duration:.1f} hours\n"
        summary += f"  • Estimated cost: {plan.estimated_cost}\n"
        summary += f"  • Activities: {len(plan.itinerary)}\n\n"
        
        # Itinerary
        summary += "🗓️ Itinerary:\n"
        for activity in plan.itinerary:
            start_time = activity.get('start_time', activity.get('time', 'Unknown'))
            summary += f"  • {start_time}: {activity['activity']} at {activity['location']}\n"
        
        # Alternatives
        if plan.alternative_suggestions:
            summary += f"\n🔄 Alternative Suggestions:\n"
            for alt in plan.alternative_suggestions:
                summary += f"  • {alt}\n"
        
        return summary
    
    def check_embeddings_status(self) -> Dict[str, Any]:
        """Check if embeddings are ready and provide status"""
        status = {
            'embeddings_ready': self._embeddings_ready,
            'embedding_service_initialized': hasattr(self.embedding_service, 'model'),
            'locations_loaded': self._locations_cache is not None,
            'total_locations': len(self._locations_cache) if self._locations_cache else 0
        }
        
        if self._locations_cache and not self._embeddings_ready:
            status['message'] = "Embeddings need to be generated. Run generate_embeddings() first."
        elif self._embeddings_ready:
            status['message'] = "Embeddings are ready! You can plan dates now."
        else:
            status['message'] = "Load locations first, then generate embeddings."
        
        return status
    
    def generate_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings for all locations"""
        if not self._locations_cache:
            locations = self._get_locations()
        else:
            locations = self._locations_cache
        
        print(f"Generating embeddings for {len(locations)} locations...")
        
        try:
            self.embedding_service.generate_embeddings(locations)
            self._embeddings_ready = True
            print("✅ Embeddings generated successfully!")
            
            return {
                'success': True,
                'total_embeddings': len(locations),
                'message': 'Embeddings generated successfully!'
            }
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to generate embeddings. Check your setup.'
            }
