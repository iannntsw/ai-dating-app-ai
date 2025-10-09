import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from .data_processor import Location

class EmbeddingService:
    """Service for generating and managing embeddings for location data"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service
        
        Args:
            model_name: Sentence-BERT model to use for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.locations = []
        self.embeddings = None
        self.embeddings_file = "ai/ai_date_planner/embeddings.pkl"
        self.index_file = "ai/ai_date_planner/faiss_index.bin"
    
    def load_model(self):
        """Load the Sentence-BERT model"""
        if self.model is None:
            print(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")

    def ensure_index_ready(self, force_rebuild: bool = False):
        """Ensure FAISS index is available by loading or building from embeddings.

        This will:
        - Load embeddings from disk if not in memory
        - Load an existing FAISS index from disk when present
        - Otherwise, build the FAISS index from current embeddings and save it
        """
        # Load embeddings if not loaded
        if self.embeddings is None or not isinstance(self.embeddings, np.ndarray):
            try:
                self.load_embeddings()
            except FileNotFoundError:
                # Nothing to do if embeddings don't exist yet
                raise

        # Load existing index if present
        if not force_rebuild and os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                return
            except Exception:
                # Fall back to rebuild
                pass

        # Build a new index from embeddings
        self.build_faiss_index(self.embeddings, force_rebuild=force_rebuild)
    
    def generate_location_text(self, location: Location) -> str:
        """
        Generate text representation of a location for embedding
        
        Args:
            location: Location object to convert to text
            
        Returns:
            Text representation of the location
        """
        text_parts = []
        
        # Add name
        text_parts.append(location.name)
        
        # Add location type
        text_parts.append(location.location_type)
        
        # Add date vibe (compatible date types)
        if location.date_vibe:
            vibe_text = " ".join(location.date_vibe)
            text_parts.append(f"suitable for {vibe_text} dates")
        
        # Add description
        if location.description:
            text_parts.append(location.description)
        
        # Add address
        if location.address:
            text_parts.append(location.address)
        
        # Add relevant metadata based on location type
        if location.location_type == "food":
            # For restaurants, add any cuisine hints from metadata
            if 'LIC_NAME' in location.metadata:
                text_parts.append(location.metadata['LIC_NAME'])
        
        elif location.location_type == "attraction":
            # For attractions, add any additional info from HTML
            if 'Description' in location.metadata:
                # Extract overview from HTML if available
                import re
                overview_match = re.search(r'<th>OVERVIEW</th>\s*<td>([^<]+)</td>', location.metadata['Description'])
                if overview_match:
                    text_parts.append(overview_match.group(1).strip())
        
        elif location.location_type == "activity":
            # For sports facilities, add facilities info
            if 'Description' in location.metadata:
                import re
                facilities_match = re.search(r'<th>FACILITIES</th>\s*<td>([^<]+)</td>', location.metadata['Description'])
                if facilities_match:
                    text_parts.append(f"Facilities: {facilities_match.group(1).strip()}")
        
        elif location.location_type == "heritage":
            # For heritage trails, add trail name
            if 'trail_name' in location.metadata:
                text_parts.append(f"Trail: {location.metadata['trail_name']}")
        
        return " | ".join(text_parts)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        🎯 SINGLE QUERY EMBEDDING (for RAG search)
        Generate embedding for a single text string (like user's search query)
        
        Used by: RAG service when user asks "romantic dinner with city views"
        Input: Single text string
        Output: Single embedding vector (384 dimensions)
        No file I/O - just computes and returns
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array representing the text embedding
        """
        self.load_model()
        return self.model.encode([text])[0]
    
    def generate_embeddings(self, locations: List[Location], force_regenerate: bool = False) -> np.ndarray:
        """
        🏗️ BULK DATABASE EMBEDDINGS (for setup/initialization)
        Generate embeddings for ALL locations in the database (31,636 locations)
        
        Used by: Setup script to create embeddings for entire location database
        Input: List of Location objects (thousands of them)
        Output: Array of embeddings + saves to embeddings.pkl file
        File I/O: Saves embeddings to disk for future use
        
        Args:
            locations: List of Location objects
            force_regenerate: Force regeneration even if embeddings exist
            
        Returns:
            Numpy array of embeddings
        """
        # Check if embeddings already exist
        if not force_regenerate and os.path.exists(self.embeddings_file):
            print("Loading existing embeddings...")
            return self.load_embeddings()
        
        print(f"Generating embeddings for {len(locations)} locations...")
        
        # Load model if not already loaded
        self.load_model()
        
        # Generate text representations
        location_texts = []
        for location in locations:
            text = self.generate_location_text(location)
            location_texts.append(text)
        
        # Generate embeddings
        print("Computing embeddings...")
        embeddings = self.model.encode(location_texts, show_progress_bar=True)
        
        # Store data
        self.locations = locations
        self.embeddings = embeddings
        
        # Save embeddings
        self.save_embeddings()
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, force_rebuild: bool = False):
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Numpy array of embeddings
            force_rebuild: Force rebuild even if index exists
        """
        # Check if index already exists
        if not force_rebuild and os.path.exists(self.index_file):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_file)
            return
        
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(self.index, self.index_file)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_embeddings(self):
        """Save embeddings and locations to file"""
        if self.embeddings is None or self.locations is None:
            print("No embeddings or locations to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
        
        # Prepare data for saving
        data = {
            'embeddings': self.embeddings,
            'locations': self.locations,
            'model_name': self.model_name
        }
        
        # Save to file
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Embeddings saved to {self.embeddings_file}")
    
    def load_embeddings(self) -> np.ndarray:
        """Load embeddings and locations from file"""
        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        with open(self.embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.locations = data['locations']
        self.model_name = data.get('model_name', self.model_name)
        
        print(f"Loaded embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def get_location_embedding(self, location_id: str) -> Optional[np.ndarray]:
        """
        🎯 GET SINGLE LOCATION EMBEDDING (for RAG similarity search)
        Retrieve pre-generated embedding for a specific location by ID
        
        Used by: RAG service during similarity search
        Input: Location ID string
        Output: Pre-generated embedding vector (384 dimensions) or None
        
        Args:
            location_id: ID of the location to get embedding for
            
        Returns:
            Numpy array of the location's embedding, or None if not found
        """
        if self.embeddings is None or self.locations is None:
            # Try to load embeddings if not already loaded
            try:
                self.load_embeddings()
            except FileNotFoundError:
                return None
        
        # Find the location index
        for i, location in enumerate(self.locations):
            if location.id == location_id:
                return self.embeddings[i]
        
        return None
    
    def similarity_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform similarity search using FAISS index
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of dictionaries with location and similarity score
        """
        if self.index is None or self.model is None:
            raise ValueError("Index or model not loaded. Call load_model() and build_faiss_index() first.")
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.locations):
                location = self.locations[idx]
                results.append({
                    'location': location,
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def get_location_by_id(self, location_id: str) -> Optional[Location]:
        """Get location by ID"""
        for location in self.locations:
            if location.id == location_id:
                return location
        return None
    
    def get_locations_by_type(self, location_type: str) -> List[Location]:
        """Get all locations of a specific type"""
        return [loc for loc in self.locations if loc.location_type == location_type]
    
    def get_locations_near_coordinates(self, lat: float, lon: float, radius_km: float = 5.0) -> List[Location]:
        """
        Get locations within a certain radius of given coordinates
        
        Args:
            lat: Latitude
            lon: Longitude  
            radius_km: Radius in kilometers
            
        Returns:
            List of locations within radius
        """
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate the great circle distance between two points on earth"""
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        nearby_locations = []
        for location in self.locations:
            distance = haversine(lon, lat, location.coordinates[0], location.coordinates[1])
            if distance <= radius_km:
                nearby_locations.append(location)
        
        return nearby_locations
