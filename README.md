# AI Dating App - AI Service

This is the AI microservice component of the AI Dating App ecosystem. It provides AI-powered features like bio generation, matching algorithms, and conversation starters using LangChain and Google Gemini.

## Architecture

This AI service works as a separate microservice that communicates with the frontend:

```
Frontend (React Native) → AI Service (FastAPI)
```

## Features

- **Bio Generation**: Creates personalized dating bios based on user interests
- **Prompt Enhancement**: Improves user responses to dating prompts
- **Lovabot**: AI dating coach with RAG-powered advice from dating articles
- **AI Date Planner**: Intelligent date planning with location-based recommendations
- **Image Quality Assessment**: AI-powered photo quality scoring using NIMA (Neural Image Assessment)
- **AI-Powered Matching**: Analyzes user compatibility (planned)
- **Conversation Starters**: Generates ice-breaker messages (planned)
- **Content Moderation**: AI-powered content filtering (planned)

## Prerequisites

- Python 3.9 or higher
- Docker (for Image Quality Assessment)
- OpenAI API key (for GPT models)
- Google Gemini API key (for embeddings)
- Access to the main NestJS backend

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-dating-app-ai
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   # Copy the example environment file
   cp .env.copy .env

   # Edit .env file and add your API keys
   # Replace the placeholder values with your actual API keys
   ```

4. **Get API Keys**

   - **OpenAI API Key**: Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Google Gemini API Key**: Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Copy the keys and paste them in your `.env` file

5. **One-Time Setup: Generate RAG Embeddings**

   **For Lovabot (Chat System):**

   ```bash
   # Run this once to process all PDFs and create embeddings for Lovabot
   python setup_lovabot_embeddings.py
   ```

   **For AI Date Planner (Location System):**

   ```bash
   # Run this once to generate embeddings for all locations and test the date planner
   python setup_date_planner_embeddings.py
   ```

   **For Vendor Activities (Dynamic Activities):**

   ```bash
   # Manual trigger to generate vendor embeddings (for testing)
   curl -X POST http://localhost:8000/api/vendor/embeddings/manual

   # Check vendor embedding status
   curl http://localhost:8000/api/vendor/embeddings/status
   ```

   This will:

   - **Lovabot**: Read all PDF files from `ai/ai_lovabot/data/` folder, process and create embeddings for dating articles, save to `ai/ai_lovabot/embeddings.pkl`
   - **Date Planner**: Load all location data from GeoJSON/KML files, generate embeddings for locations, save to `ai/ai_date_planner/embeddings/` folder
   - **Vendor Activities**: Fetch vendor activities from MongoDB, generate embeddings, save to `ai/ai_date_planner/embeddings_vendors.pkl` and `faiss_vendors.bin`
   - **Cron Job**: Automatically regenerates vendor embeddings daily at 12:00 AM
   - Both systems will be ready to use after running these setup scripts

6. **Set up Image Quality Assessment (NIMA)**

   ```bash
   # Navigate to the image quality assessment directory
   cd ai/image-quality-assessment

   # Build the Docker image for NIMA (Neural Image Assessment)
   docker build -t nima-cpu . -f Dockerfile.cpu
   ```

   This will:

   - Build a Docker image with TensorFlow 2.0.0 and NIMA dependencies
   - Install required system packages (bzip2, g++, git, graphviz, etc.)
   - Install Python dependencies (scikit-learn, pillow, nose)
   - Create a ready-to-use image quality assessment container

## Running the Service

```bash
# Start the FastAPI server using the new fastapi dev command, this allows FastAPI to listen on all ports
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://YOUR_IP_ADDRESS:8000`

## API Documentation

Once running, visit:

- **Swagger UI**: `http://127.0.0.1:8000/docs` (recommended)
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## API Endpoints

### POST /ai/bio

Generate personalized dating bios based on user interests.

**Request:**

```json
{
  "bio_interests": ["cooking", "hiking", "photography", "reading"]
}
```

**Response:**

```json
[
  "I love cooking and hiking on weekends. Looking for someone to share adventures with!",
  "Photography and cooking are my passions. Let's explore the world together through food and lens.",
  "Adventure seeker who loves hiking and capturing moments. Ready to create memories with the right person!"
]
```

### POST /ai/prompts

Enhance user responses to dating prompts.

**Request:**

```json
{
  "question": "What's your ideal Sunday?",
  "answer": "Sleeping in"
}
```

**Response:**

```json
"Perfect lazy Sunday vibes! I love sleeping in and having a slow morning with coffee and maybe some Netflix. What's your go-to lazy day routine?"
```

### POST /ai/lovabot

Chat with Lovabot - your AI dating coach with access to dating articles and advice.

**Request:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How do I start a conversation on a dating app?"
    },
    {
      "role": "assistant",
      "content": "Start with something specific from their profile..."
    },
    { "role": "user", "content": "What if they don't respond?" }
  ]
}
```

**Response:**

```json
{
  "answer": "Don't worry if they don't respond immediately! People get busy. Try a follow-up message after a few days, or focus on other matches. Remember, it's not personal - keep the positive energy! 💕"
}
```

### POST /ai/plan-date

Plan a complete date itinerary using AI-powered location recommendations.

### POST /ai/assess-image-quality

Assess the quality of user profile photos using NIMA (Neural Image Assessment).

**Request:**

```json
{
  "image_url": "https://example.com/profile-photo.jpg"
}
```

**Response:**

```json
{
  "quality_score": 7.2,
  "technical_score": 6.8,
  "aesthetic_score": 7.6,
  "recommendations": [
    "Consider better lighting",
    "Try a different angle",
    "Ensure the image is in focus"
  ]
}
```

**Request:**

```json
{
  "start_time": "10:00",
  "end_time": "17:00",
  "start_latitude": 1.3521,
  "start_longitude": 103.8198,
  "interests": ["food", "culture", "nature"],
  "budget_tier": "$$",
  "date_type": "romantic",
  "preferred_location_types": ["food", "attraction", "activity", "heritage"],
  "user_query": "romantic date with city views"
}
```

**Response:**

```json
{
  "success": true,
  "itinerary": [
    {
      "time": "10:00",
      "activity": "Lunch",
      "location": "Marina Bay Sands Food Court",
      "type": "food",
      "duration": 1.5,
      "description": "Enjoy lunch at Marina Bay Sands Food Court..."
    },
    {
      "time": "11:30",
      "activity": "Cultural Visit",
      "location": "ArtScience Museum",
      "type": "attraction",
      "duration": 3.0,
      "description": "Explore ArtScience Museum..."
    }
  ],
  "total_duration": 7.0,
  "estimated_cost": "$50-$100 per person",
  "summary": "Your 7.0-hour romantic date: • 10:00: Lunch at Marina Bay Sands Food Court • 11:30: Cultural Visit at ArtScience Museum",
  "alternative_suggestions": [
    "Alternative food: ION Orchard Food Hall",
    "Alternative attraction: Gardens by the Bay"
  ],
  "processing_stats": {
    "total_locations": 10000,
    "filtered_locations": 200,
    "relevant_locations": 50,
    "final_activities": 2,
    "embeddings_ready": true
  }
}
```

## Integration with Main Backend

This AI service is designed to be called by the main NestJS backend:

1. **NestJS Backend** receives user requests
2. **Calls AI Service** via HTTP requests
3. **AI Service** processes with LangChain + Gemini
4. **Returns results** to NestJS backend
5. **NestJS** sends response to frontend

### Example NestJS Integration:

```typescript
// In your NestJS service
async generateBio(interests: string[]) {
  const response = await this.httpService.post(
    'http://127.0.0.1:8000/bio',
    { bio_interests: interests }
  ).toPromise();
  return response.data;
}
```

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional for tracing
LANGSMITH_PROJECT=your_project_name        # Optional for tracing
```

### Project Structure

```
ai-dating-app-ai/
├── main.py                              # FastAPI application
├── setup_lovabot_embeddings.py          # One-time setup script for Lovabot RAG
├── setup_date_planner_embeddings.py     # One-time setup script for Date Planner RAG
├── requirements.txt                     # Python dependencies
├── .env.copy                           # Environment template
├── .env                                # Your environment variables (ignored by git)
├── .gitignore
├── README.md
└── ai/
    ├── profile_management/
    │   ├── ai_profile_management.py     # AI model initialization
    │   ├── ai_bio_generator.md          # Bio generation prompts
    │   └── ai_prompt_generator.md       # Prompt enhancement prompts
    ├── ai_lovabot/
    │   ├── ai_lovabot.py                # Lovabot with RAG functionality
    │   ├── ai_lovabot_instructions.md   # Lovabot system prompt
    │   ├── embeddings.pkl               # Pre-processed embeddings (generated)
    │   └── data/                        # Dating articles (PDFs)
    │       └── Article 1 dating.pdf
    └── ai_date_planner/
        ├── ai_date_planner.py           # Main date planning orchestrator
        ├── data_processor.py            # Location data processing
        ├── embedding_service.py         # Location embeddings service
        ├── rule_engine.py               # Rule-based filtering
        ├── rag_service.py               # RAG-based location retrieval
        ├── embeddings/                  # Location embeddings (generated)
        └── data/                        # Location data (GeoJSON/KML)
            ├── EatingEstablishments.geojson
            ├── HeritageTrails.kml
            ├── SportSGSportFacilitiesGEOJSON.geojson
            └── TouristAttractions.geojson
    └── image-quality-assessment/
        ├── Dockerfile.cpu               # CPU-based Docker configuration
        ├── Dockerfile.gpu               # GPU-based Docker configuration
        ├── src/                         # NIMA source code
        │   ├── requirements.txt         # Python dependencies
        │   ├── trainer/                 # Training modules
        │   ├── evaluater/               # Evaluation modules
        │   └── utils/                   # Utility functions
        ├── entrypoints/                 # Docker entrypoint scripts
        ├── models/                      # Pre-trained NIMA models
        └── data/                        # Training and test datasets
```

### Adding New AI Features

1. Create new prompt files in `ai/profile_management/` or create new folder
2. Add new endpoints in `main.py`
3. Update this README with new API documentation

## Troubleshooting

### Common Issues

1. **API key errors**: Verify your Google Gemini API key is correct
2. **Port conflicts**: Change the port in the uvicorn command if 8000 is taken
3. **Docker build errors**: If you encounter issues building the NIMA image, try:
   - `docker system prune -f` to clean up Docker cache
   - Ensure Docker Desktop is running
   - Check that you have sufficient disk space (image is ~1.6GB)

### Logs

The service uses LangSmith for tracing. Check your LangSmith dashboard for detailed AI interaction logs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

This project uses the Image Quality Assessment model for photo quality scoring. Please cite the following if you use this in your research:

```bibtex
@misc{idealods2018imagequalityassessment,
  title={Image Quality Assessment},
  author={Christopher Lennan and Hao Nguyen and Dat Tran},
  year={2018},
  howpublished={\url{https://github.com/idealo/image-quality-assessment}},
}
```

## License

This project is part of the AI Dating App ecosystem.
