# System Architecture

## Overview

The XAI Sentiment Analysis System is built using a microservices architecture with three main components:

1. **Dashboard Service** - Web interface and user management
2. **XAI Service** - Sentiment analysis and visualization generation
3. **AI Assistant Service** - RAG-powered chat functionality

## Component Details

### Dashboard Service (Port 3000)

**Purpose**: Web interface for user interaction and file management

**Technologies**:
- Flask web framework
- HTML/CSS/JavaScript frontend
- Session-based authentication
- File upload handling

**Key Features**:
- User authentication and session management
- File upload for data and models
- Results visualization
- Chat interface integration

### XAI Service (Port 8000)

**Purpose**: Core sentiment analysis and XAI visualization generation

**Technologies**:
- Flask API
- FinBERT for sentiment analysis
- Matplotlib/Seaborn for visualizations
- LIME for model explanations
- Transformers for attention analysis

**Key Features**:
- Financial text sentiment analysis
- LIME explanations for predictions
- Attention analysis for transformer models
- Word sentiment associations
- Data statistics and insights

### AI Assistant Service (Port 8001)

**Purpose**: RAG-powered chat system for intelligent Q&A

**Technologies**:
- Flask API
- OpenAI GPT-3.5-turbo
- Vector database (in-memory)
- OpenAI embeddings

**Key Features**:
- Context-aware responses
- Vector similarity search
- User-specific data isolation
- Fallback mechanisms

## Data Flow

```
User Upload → Dashboard → XAI Service → Analysis → AI Assistant → Vector DB
     ↓              ↓           ↓           ↓           ↓
  File Storage → Session → FinBERT → Visualizations → RAG Chat
```

## Storage Architecture

### Shared Volume Structure
```
shared_volume/
├── uploads/          # User uploaded files
├── models/           # Trained models
├── results/          # Analysis results
├── images/           # Generated visualizations
└── vector_db/        # Vector database storage
```

### User Isolation
- Each user has isolated data storage
- User-specific result directories
- Separate vector collections per user
- Session-based access control

## Security Considerations

### Current Implementation
- Session-based authentication
- User data isolation
- Input validation
- Secure file uploads

### Production Recommendations
- Database-backed user management
- JWT token authentication
- HTTPS enforcement
- Rate limiting
- API key rotation

## Scalability

### Horizontal Scaling
- Stateless services can be replicated
- Load balancer for multiple instances
- Shared volume can be replaced with cloud storage

### Vertical Scaling
- Increase container resources
- Optimize model inference
- Cache frequently accessed data

## Monitoring and Logging

### Health Checks
- `/health` endpoints for each service
- Docker health checks
- Service dependency monitoring

### Logging
- Structured logging with timestamps
- Error tracking and alerting
- Performance metrics collection 