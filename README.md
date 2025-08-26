# FoodSnap

AI-powered food image analysis pipeline that transforms food photos into detailed recipes, ingredient lists, and nutritional information.

## Overview

FoodSnap leverages state-of-the-art vision and language models to provide comprehensive food analysis from a single image. The system combines BLIP for accurate image understanding with Llama-3.2-3B-Instruct for structured culinary analysis, delivering professional-grade results in seconds.

## Features

- **Instant Food Recognition**: Upload any food image for immediate analysis
- **Detailed Recipe Generation**: Get complete cooking instructions with ingredients and measurements
- **Nutritional Analysis**: Comprehensive breakdown of calories, macronutrients, etc.
- **Professional Culinary Insights**: Dish history, cuisine classification, and cooking tips
- **Smart Caching**: Optimized performance through intelligent result caching
- **Demo Mode**: Built-in sample images for quick testing

## Technical Architecture

### Backend
- **Framework**: FastAPI with async/await for high-performance API endpoints
- **Models**: 
  - BLIP (Salesforce/blip-image-captioning-large) for vision understanding
  - Llama-3.2-3B-Instruct for structured food analysis
- **Deployment**: Modal serverless platform with GPU acceleration
- **Caching**: In-memory cache with TTL for optimized repeated queries

### Frontend
- **Technologies**: Vanilla JavaScript, HTML5, CSS3
- **Design**: Responsive interface with modern aesthetics
- **Features**: Drag-and-drop upload, real-time processing feedback, demo gallery

## Performance

- **Average Response Time**: 10 seconds for complete analysis
- **GPU Optimization**: CUDA-accelerated inference on Modal
- **Concurrent Handling**: Supports multiple simultaneous requests

## Installation

### Prerequisites
- Python 3.11+
- Modal account (free tier available)
- Hugging Face account for model access

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/revilo440/foodsnap.git
   cd foodsnap
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Modal and Hugging Face credentials:
   ```bash
   modal setup
   modal secret create huggingface-secret HF_TOKEN=your_token_here
   ```

4. Deploy to Modal:
   ```bash
   modal deploy app.py
   ```

5. Start the local web server:
   ```bash
   cd web && python -m http.server 8080
   ```

6. Open http://localhost:8080 in your browser

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

## API Documentation

### POST /analyze
Analyzes a food image and returns structured culinary data.

**Request:**
- Body: `multipart/form-data` with image file
- Supported formats: JPEG, PNG, WebP

**Response:**
```json
{
  "success": true,
  "data": {
    "caption": "Grilled salmon with roasted vegetables",
    "confidence": 0.92,
    "dish_name": "Grilled Salmon with Roasted Vegetables",
    "cuisine": "Mediterranean",
    "description": "...",
    "ingredients": [...],
    "instructions": [...],
    "nutrition": {...},
    "history": "...",
    "tips": [...]
  },
  "timing": {
    "total_time": 3.45
  }
}
```

## Project Structure

```
foodsnap/
├── app.py                 # Modal backend with FastAPI endpoints
├── DEPLOYMENT_GUIDE.md    # Quick deployment instructions
├── README.md              # This file
└── web/
    ├── index.html        # Main application interface
    ├── app.js            # Frontend logic and API integration
    ├── demo.js           # Demo mode functionality
    ├── error-handler.js  # Error handling utilities
    └── styles.css        # UI styling
```

## Development

### Local Testing
Run the backend locally with Modal's development server:
```bash
modal serve app.py
```

### Adding Demo Images
Place new demo images in `web/demo-images/` and update the demo gallery in `app.js`.

### Model Configuration
Adjust model parameters in `app.py`:
- Temperature: Controls creativity (default: 0.3)
- Max tokens: Response length limit (default: 800)
- Top-p: Nucleus sampling parameter (default: 0.9)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

Oliver Murcko

## Acknowledgments

- Hugging Face for model hosting and transformers library
- Modal for serverless GPU infrastructure
- Salesforce Research for BLIP model
- Meta AI for Llama models
