# XAI Sentiment Analysis System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

A comprehensive **Explainable AI (XAI) system** for financial sentiment analysis with **RAG-powered AI assistant** capabilities. Built with Docker microservices architecture.

## 🚀 Features

- **📊 Advanced Sentiment Analysis**: FinBERT-powered financial text analysis
- **🔍 Explainable AI Visualizations**: LIME, Attention Analysis, Word Sentiment Associations
- **💬 Intelligent AI Assistant**: RAG-powered chat with context-aware responses
- **🔐 User Authentication**: Secure multi-user system with data isolation
- **📈 Data Statistics**: Comprehensive data insights and visualizations
- **🐳 Docker Architecture**: Microservices with shared volume storage

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   XAI Service   │    │  AI Assistant   │
│   (Port 3000)   │◄──►│   (Port 8000)   │◄──►│   (Port 8001)   │
│                 │    │                 │    │                 │
│ • Web Interface │    │ • FinBERT Model │    │ • RAG Chat      │
│ • File Upload   │    │ • Visualizations│    │ • Vector DB     │
│ • Results View  │    │ • Data Analysis │    │ • OpenAI API    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Quick Start

### Prerequisites
- Docker and Docker Compose
- At least 4GB RAM
- OpenAI API key (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/xai-sentiment-analysis-system.git
   cd xai-sentiment-analysis-system
   ```

2. **Set up OpenAI API key (optional)**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Start the system**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the dashboard**
   - Open http://localhost:3000
   - Login: `admin` / `password123`

## 📖 Usage

### 1. Upload Financial Data
- Supported formats: CSV, JSON, Excel, Parquet
- Upload financial news, earnings reports, or market commentary

### 2. Generate Sentiment Analysis
- System automatically processes text with FinBERT
- Generates comprehensive sentiment visualizations

### 3. Explore XAI Visualizations
- **LIME Analysis**: Word-level importance for predictions
- **Attention Analysis**: Transformer attention patterns
- **Word Sentiment Associations**: Top positive/negative words
- **Data Statistics**: Comprehensive data insights

### 4. Chat with AI Assistant
- Ask questions about your analysis results
- Get context-aware responses powered by RAG
- Example questions:
  - "What are the top 3 negative words?"
  - "How does the model make predictions?"
  - "Describe the attention analysis"

## ⚙️ Configuration

### Environment Variables
```bash
# Required for OpenAI integration
OPENAI_API_KEY=your-api-key

# Optional configurations
DEBUG=false
LOG_LEVEL=INFO
```

### Docker Services
- **Dashboard**: Web interface (Port 3000)
- **XAI Service**: Analysis engine (Port 8000)
- **AI Assistant**: RAG chat system (Port 8001)

## 🔌 API Endpoints

### Dashboard API
- `POST /api/upload-data` - Upload data files
- `POST /api/upload-model` - Upload model files
- `GET /api/get-results` - Get analysis results
- `POST /api/chat` - Chat with AI assistant

### XAI Service API
- `POST /ingest` - Ingest uploaded data
- `POST /analyze` - Analyze uploaded model
- `POST /data-statistics` - Generate data insights

### AI Assistant API
- `POST /chat` - Handle chat questions
- `GET /results/{user_id}` - Get user results
- `DELETE /clear-user-data/{user_id}` - Clear user data

## 🧪 Testing

### Sample Data
The system includes sample financial sentiment data for testing:
- Financial news articles with sentiment scores
- Multiple asset coverage (stocks, crypto, commodities)
- Balanced positive/negative distribution

### Test Questions
```bash
# Test AI assistant
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top negative words?", "user_id": "admin"}'
```

## 🔒 Security

### Current Implementation
- Session-based authentication
- User data isolation
- Secure file uploads
- Input validation

### Production Recommendations
- Database-backed user management
- JWT token authentication
- HTTPS enforcement
- Rate limiting
- API key rotation

## 🚀 Deployment

### Local Development
```bash
docker-compose up --build
```

### Production Deployment
```bash
# Set production environment variables
export PRODUCTION=true
export SECRET_KEY=your-secret-key

# Deploy with production settings
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FinBERT](https://github.com/ProsusAI/finbert) for financial sentiment analysis
- [OpenAI](https://openai.com/) for RAG capabilities
- [Hugging Face](https://huggingface.co/) for transformer models
- [Docker](https://www.docker.com/) for containerization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/xai-sentiment-analysis-system/issues)
- **Documentation**: See [docs/](docs/) folder
- **Email**: your-email@example.com

---

**Made with ❤️ for Explainable AI and Financial Analysis**
