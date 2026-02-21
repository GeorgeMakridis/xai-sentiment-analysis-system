# Quick Start Guide

## Option 1: Docker Setup (Recommended)

### Prerequisites
- Docker Desktop installed
- At least 4GB RAM available

### Steps
1. Run the setup script:
   ```bash
   ./setup.sh
   ```

2. Or manually:
   ```bash
   docker compose up --build -d
   ```

3. Access the dashboard at http://localhost:3000
4. Login with:
   - Username: `admin`
   - Password: `password123`

## Option 2: Manual Setup (For Development)

### Prerequisites
- Python 3.9+
- pip

### Steps

1. **Setup Dashboard Service**
   ```bash
   cd dashboard
   pip install -r requirements.txt
   python app.py
   ```

2. **Setup XAI Service** (in new terminal)
   ```bash
   cd xai_service
   pip install -r requirements.txt
   python app.py
   ```

3. **Setup AI Outputs Service** (in new terminal)
   ```bash
   cd ai_outputs
   pip install -r requirements.txt
   python app.py
   ```

4. Access the dashboard at http://localhost:3000

## Demo Mode

If you don't have your own data or model:

1. Login to the dashboard
2. The system will automatically generate sample data and model
3. You'll see XAI visualizations for a loan approval dataset
4. Use the chat to ask questions about the results

## Sample Questions to Try

- "What are the most important features in this model?"
- "How does the model make predictions?"
- "What patterns does the model identify?"
- "Are there any biases in the model?"
- "How well does the model perform?"

## Troubleshooting

### Port Issues
If you get port conflicts:
```bash
# Check what's using the ports
lsof -i :3000
lsof -i :8000
lsof -i :8001

# Stop conflicting services or change ports in docker-compose.yml
```

### Memory Issues
If containers fail to start:
- Increase Docker memory allocation (8GB+ recommended)
- Close other applications to free up memory

### File Upload Issues
- Ensure file formats are supported
- Check file size (max 100MB recommended)
- Try the sample data provided

## Support

- Check the main README.md for detailed documentation
- View container logs: `docker compose logs`
- Restart services: `docker compose restart` 