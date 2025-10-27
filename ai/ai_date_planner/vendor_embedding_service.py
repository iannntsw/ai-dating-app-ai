import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pymongo import MongoClient
from datetime import datetime
import hashlib
import json

class VendorEmbeddingService:
    """Service for generating and managing vendor activity embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        # Vendor-specific paths
        self.vendor_embeddings_file = "ai/ai_date_planner/embeddings_vendors.pkl"
        self.vendor_index_file = "ai/ai_date_planner/faiss_vendors.bin"
        
        # MongoDB connection
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.mongo_db = os.getenv('MONGO_DB', 'aiDatingApp')
        self.client = None
        self.db = None
        
        # Vendor data cache
        self.vendor_locations = []
        self.vendor_embeddings = None
        self.vendor_index = None
    
    def connect_to_mongodb(self):
        """Connect to MongoDB"""
        if self.client is None:
            try:
                print(f"Connecting to MongoDB: {self.mongo_uri}/{self.mongo_db}")
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client[self.mongo_db]
                print("✅ MongoDB connected successfully")
            except Exception as e:
                print(f"❌ MongoDB connection failed: {e}")
                raise
    
    def disconnect_from_mongodb(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.client = None
            print("✅ MongoDB disconnected")
    
    def fetch_vendor_activities(self) -> List[Dict[str, Any]]:
        """Fetch active vendor activities from MongoDB"""
        if self.db is None:
            raise Exception("MongoDB not connected")
        
        try:
            # Fetch activities that are active and not expired
            # Include activities that haven't started yet, as long as they're not expired
            activities = list(self.db.activities.find({
                'isActive': True,
                '$or': [
                    # No endDate - always available
                    {'endDate': {'$exists': False}},
                    # Has endDate but it's today or later (not expired)
                    {'endDate': {'$exists': True, '$gte': datetime.now()}}
                ]
            }))
            print(f"✅ Fetched {len(activities)} active vendor activities")
            
            # Debug: Show which activities were fetched
            for activity in activities:
                print(f"  📋 Fetched: {activity.get('title', 'Untitled')} (Active: {activity.get('isActive')}, EndDate: {activity.get('endDate')})")
            
            return activities
        except Exception as e:
            print(f"❌ Error fetching vendor activities: {e}")
            raise
    
    def convert_vendor_activity_to_location(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vendor activity to location format compatible with existing system"""
        try:
            # Generate rich unique ID
            activity_id = f"vendor_{activity['_id']}"
            
            # Map category to date vibe
            category_to_vibe = {
                'Adventurous': ['adventurous'],
                'Heritage': ['cultural'],
                'Romantic': ['romantic'],
                'Casual': ['casual']
            }
            
            # Map activityType to location_type
            activity_type_to_location_type = {
                'Food': 'food',
                'Sports': 'activity',
                'Workshop': 'activity',
                'Attraction Visit': 'attraction'
            }
            
            category = activity.get('category', 'Casual')
            activity_type = activity.get('activityType', 'Attraction Visit')
            
            # Debug: Log the mapping
            mapped_location_type = activity_type_to_location_type.get(activity_type, 'attraction')
            print(f"  🔍 Vendor: '{activity.get('title')}' | activityType: '{activity_type}' → location_type: '{mapped_location_type}'")
            
            # Create location object
            location = {
                'id': activity_id,
                'name': activity.get('title', 'Untitled Activity'),
                'location_type': mapped_location_type,
                'coordinates': activity.get('coordinates', {}).get('coordinates', [0, 0]),
                'address': activity.get('location', ''),
                'description': activity.get('description', ''),
                'metadata': {
                    'vendorId': str(activity['_id']),
                    'price': activity.get('price', 0),
                    'durationMinutes': activity.get('durationMinutes', 120),
                    'startDate': activity.get('startDate'),
                    'endDate': activity.get('endDate'),
                    'imageUrl': activity.get('imageUrl'),
                    'isVendor': True,
                    'isActive': activity.get('isActive', True),
                    'category': category,
                    'activityType': activity_type
                },
                'date_vibe': category_to_vibe.get(category, ['casual'])
            }
            
            return location
            
        except Exception as e:
            print(f"❌ Error converting vendor activity: {e}")
            return None
    
    def generate_vendor_embeddings(self, force_regenerate: bool = False) -> np.ndarray:
        """Generate embeddings for vendor activities"""
        # Check if embeddings already exist
        if not force_regenerate and os.path.exists(self.vendor_embeddings_file):
            print("Loading existing vendor embeddings...")
            return self.load_vendor_embeddings()
        
        print("🏗️ Generating VENDOR embeddings...")
        
        # Connect to MongoDB and fetch activities
        self.connect_to_mongodb()
        activities = self.fetch_vendor_activities()
        
        if not activities:
            print("⚠️ No vendor activities found")
            return np.array([])
        
        # Convert activities to locations with deduplication
        vendor_locations = []
        seen_vendor_ids = set()  # Track unique vendor activities
        
        for activity in activities:
            # Use vendor ID + title as unique key to prevent duplicates
            vendor_key = f"{activity.get('vendorId', 'unknown')}_{activity.get('title', 'untitled')}"
            
            if vendor_key in seen_vendor_ids:
                print(f"  🔄 Skipping duplicate vendor activity: {activity.get('title', 'Untitled')}")
                continue
                
            seen_vendor_ids.add(vendor_key)
            
            location = self.convert_vendor_activity_to_location(activity)
            if location:
                vendor_locations.append(location)
        
        if not vendor_locations:
            print("⚠️ No valid vendor locations after conversion")
            return np.array([])
        
        print(f"📊 Processing {len(vendor_locations)} vendor locations...")
        
        # Load model
        if self.model is None:
            print(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded successfully")
        
        # Generate embeddings
        embedding_texts = []
        for location in vendor_locations:
            # Create rich embedding text
            text_parts = [
                location['name'],
                location.get('description', ''),
                location.get('address', ''),
                location['metadata'].get('category', ''),
                location['metadata'].get('activityType', ''),
                ' '.join(location.get('date_vibe', []))
            ]
            embedding_text = ' '.join(filter(None, text_parts))
            embedding_texts.append(embedding_text)
        
        print(f"🔤 Generating embeddings for {len(embedding_texts)} vendor activities...")
        embeddings = self.model.encode(embedding_texts, convert_to_numpy=True)
        
        # Store data
        self.vendor_locations = vendor_locations
        self.vendor_embeddings = embeddings
        
        # Save embeddings
        self.save_vendor_embeddings()
        
        print(f"✅ Vendor embeddings generated: {embeddings.shape}")
        return embeddings
    
    def save_vendor_embeddings(self):
        """Save vendor embeddings and locations to file"""
        if self.vendor_embeddings is None:
            print("No vendor embeddings to save")
            return
        
        os.makedirs(os.path.dirname(self.vendor_embeddings_file), exist_ok=True)
        
        data = {
            'embeddings': self.vendor_embeddings,
            'locations': self.vendor_locations,
            'model_name': self.model_name,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(self.vendor_embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Vendor embeddings saved to {self.vendor_embeddings_file}")
    
    def load_vendor_embeddings(self) -> np.ndarray:
        """Load vendor embeddings and locations from file"""
        if not os.path.exists(self.vendor_embeddings_file):
            raise FileNotFoundError(f"Vendor embeddings file not found: {self.vendor_embeddings_file}")
        
        with open(self.vendor_embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.vendor_embeddings = data['embeddings']
        self.vendor_locations = data['locations']
        
        print(f"✅ Loaded vendor embeddings: {self.vendor_embeddings.shape}")
        return self.vendor_embeddings
    
    def build_vendor_faiss_index(self) -> faiss.Index:
        """Build FAISS index for vendor embeddings"""
        if self.vendor_embeddings is None:
            raise ValueError("No vendor embeddings available. Generate embeddings first.")
        
        print("🏗️ Building vendor FAISS index...")
        
        # Create FAISS index
        dimension = self.vendor_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.vendor_embeddings)
        
        # Add embeddings to index
        index.add(self.vendor_embeddings.astype('float32'))
        
        # Save index
        os.makedirs(os.path.dirname(self.vendor_index_file), exist_ok=True)
        faiss.write_index(index, self.vendor_index_file)
        
        self.vendor_index = index
        print(f"✅ Vendor FAISS index built and saved: {index.ntotal} vectors")
        return index
    
    def load_vendor_faiss_index(self) -> faiss.Index:
        """Load vendor FAISS index from file"""
        if not os.path.exists(self.vendor_index_file):
            raise FileNotFoundError(f"Vendor FAISS index not found: {self.vendor_index_file}")
        
        index = faiss.read_index(self.vendor_index_file)
        self.vendor_index = index
        
        print(f"✅ Loaded vendor FAISS index: {index.ntotal} vectors")
        return index
    
    def search_vendor_locations(self, query_text: str, k: int = 20) -> List[Dict[str, Any]]:
        """Search vendor locations using FAISS"""
        if self.vendor_index is None:
            try:
                self.load_vendor_faiss_index()
            except FileNotFoundError:
                print("⚠️ Vendor FAISS index not found")
                return []
        
        if self.vendor_embeddings is None:
            try:
                self.load_vendor_embeddings()
            except FileNotFoundError:
                print("⚠️ Vendor embeddings not found")
                return []
        
        # Load model if not loaded
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vendor_index.search(query_embedding.astype('float32'), k)
        
        # Return results with deduplication
        results = []
        seen_location_ids = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.vendor_locations):
                location = self.vendor_locations[idx]
                location_id = location['id']
                
                # Skip if already seen (prevent duplicates from FAISS)
                if location_id in seen_location_ids:
                    continue
                    
                seen_location_ids.add(location_id)
                results.append({
                    'location': location,
                    'score': float(score)
                })
        
        return results
    
    def get_vendor_location_by_id(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get vendor location by ID"""
        for location in self.vendor_locations:
            if location['id'] == location_id:
                return location
        return None
