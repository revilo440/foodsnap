# FoodSnap - Quick Deployment Guide

## Prerequisites

1. Install Modal CLI: `pip install modal`
2. Login to Modal: `modal token set`
3. Get Hugging Face access to Llama-3.2-3B-Instruct

## Setup & Deploy

### 1. Configure Hugging Face Token
```bash
# Get token from https://huggingface.co/settings/tokens
modal secret create huggingface-secret HUGGING_FACE_TOKEN=<your-token>
```

### 2. Deploy to Modal
```bash
cd foodsnap
modal deploy app.py
```

### 3. Update Frontend
Copy your API endpoint from deployment output and update `web/app.js`:
```javascript
const API_ENDPOINT = 'https://[your-username]--foodsnap-fastapi-app.modal.run';
```

### 4. Test
Open `web/index.html` and upload a food image.

## Common Issues

**"Secret not found"** → Run step 1 again  
**"Model not found"** → Request access at Hugging Face model page  
**"Timeout"** → Normal for first run, subsequent calls are faster  

## Support

- Modal docs: https://modal.com/docs
- Check logs: `modal logs foodsnap`
- View dashboard: https://modal.com/apps

