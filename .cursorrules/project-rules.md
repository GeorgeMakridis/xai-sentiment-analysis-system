# XAI Sentiment Analysis System - Cursor Rules

## Project Overview
This is a microservices-based Explainable AI (XAI) system for financial sentiment analysis with RAG-powered AI assistant capabilities.

## Architecture
- **Dashboard Service**: Flask web interface (Port 3001)
- **XAI Service**: Analysis engine with FinBERT (Port 8000)
- **AI Outputs Service**: RAG chat system (Port 8002)

## Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for functions and classes
- Use meaningful variable names

## Service Communication
- Services communicate via HTTP REST APIs
- Shared volume at `/app/shared_data` for file storage
- Environment variables for configuration

## Security
- Never commit API keys or secrets
- Use environment variables for sensitive data
- Reference `.env.example` for required variables

## Testing
- Test files in `tests/` directory
- Run tests before committing changes

## File Organization
- Documentation in `docs/` directory
- Test files in `tests/` directory
- Utility scripts in `scripts/` directory
- Service code in respective service directories (`dashboard/`, `xai_service/`, `ai_outputs/`)

## Docker
- Use docker-compose.yml for orchestration
- Services share volume at `shared_volume/`
- Environment variables passed via docker-compose

## Dependencies
- Python 3.9+
- Flask for web services
- Docker and Docker Compose for deployment
- OpenAI API for RAG functionality (optional)
