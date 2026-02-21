# RAG (Retrieval-Augmented Generation) System Setup

## 🎯 Overview

This project now includes a sophisticated RAG (Retrieval-Augmented Generation) system that enhances the conversational interface with:

- **Vector Database**: Stores embeddings of analysis results and visualizations
- **OpenAI Integration**: Uses GPT for intelligent, context-aware responses
- **Per-User Authorization**: Isolated vector collections for each user
- **Image Storage**: Saves figures to shared volume with metadata
- **Fallback Mechanism**: Works without OpenAI API key

## 🚀 Features

### **Intelligent Context Retrieval**
- Automatically extracts context from analysis results
- Stores model information, performance metrics, and feature importance
- Creates embeddings for semantic search
- Retrieves relevant context for each question

### **OpenAI-Powered Responses**
- Uses GPT-3.5-turbo for natural language generation
- Context-aware responses based on actual analysis data
- Specialized prompts for XAI explanations
- Fallback responses when OpenAI is unavailable

### **User Authorization**
- Per-user vector collections
- Isolated data storage
- Secure access control
- Data cleanup capabilities

### **Image Management**
- Saves visualizations to shared volume
- User-specific image folders
- Metadata tracking for each image
- Automatic cleanup on user deletion

## 🔧 Setup Instructions

### 1. **OpenAI API Key Setup (Optional but Recommended)**

#### Option A: Environment Variable
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Option B: Docker Environment
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

#### Option C: Docker Compose Override
Add to your `docker-compose.yml`:
```yaml
ai_outputs:
  environment:
    - OPENAI_API_KEY=your-openai-api-key-here
```

### 2. **Get OpenAI API Key**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Copy the key to your environment

### 3. **Rebuild and Start Services**
```bash
# Rebuild the AI outputs service
docker-compose build ai_outputs

# Start all services
docker-compose up -d
```

## 📊 How It Works

### **1. Data Ingestion Pipeline**
```
Analysis Results → Context Extraction → Vector Embeddings → Storage
```

1. **Analysis Results**: XAI service generates analysis with visualizations
2. **Context Extraction**: System extracts key information:
   - Model type and features
   - Performance metrics
   - Data summary
   - Feature importance
   - Visualization descriptions
3. **Vector Embeddings**: OpenAI embeddings for semantic search
4. **Storage**: User-specific vector collections

### **2. RAG Response Generation**
```
User Question → Semantic Search → Context Retrieval → GPT Response
```

1. **User Question**: Natural language query about analysis
2. **Semantic Search**: Find relevant documents using embeddings
3. **Context Retrieval**: Get top-k most relevant contexts
4. **GPT Response**: Generate intelligent response with context

### **3. User Authorization Flow**
```
User Login → Session Management → Isolated Vector Access → Secure Responses
```

## 🎨 Usage Examples

### **Basic Questions**
```bash
# Model information
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What type of model is this?", "user_id": "admin"}'

# Feature importance
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the most important features?", "user_id": "admin"}'

# Performance metrics
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How well does the model perform?", "user_id": "admin"}'
```

### **Advanced Questions**
```bash
# Model behavior
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the model make predictions?", "user_id": "admin"}'

# Data insights
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What patterns does the model identify?", "user_id": "admin"}'

# Visualization information
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What visualizations are available?", "user_id": "admin"}'
```

## 🔍 API Endpoints

### **Store Results**
```bash
POST /store-results
Content-Type: application/json

{
  "user_id": "user123",
  "model_info": {...},
  "performance_metrics": {...},
  "data_summary": {...},
  "images": [...]
}
```

### **Chat**
```bash
POST /chat
Content-Type: application/json

{
  "question": "What type of model is this?",
  "user_id": "user123"
}
```

### **Get Results Summary**
```bash
GET /results/{user_id}
```

### **Clear User Data**
```bash
DELETE /clear-user-data/{user_id}
```

## 🛡️ Security Features

### **User Isolation**
- Each user has their own vector collection
- No cross-user data access
- Secure session management

### **Data Privacy**
- User-specific image storage
- Isolated metadata tracking
- Automatic data cleanup

### **API Security**
- Input validation
- Error handling
- Rate limiting (can be added)

## 🔧 Configuration Options

### **Environment Variables**
```bash
# Required for OpenAI integration
OPENAI_API_KEY=your-api-key

# Optional configurations
DEBUG=false
LOG_LEVEL=INFO
```

### **Vector Database Settings**
```python
# In ai_outputs/app.py
TOP_K_RESULTS = 5  # Number of relevant documents to retrieve
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model
GPT_MODEL = "gpt-3.5-turbo"  # OpenAI chat model
MAX_TOKENS = 500  # Maximum response length
TEMPERATURE = 0.7  # Response creativity (0.0-1.0)
```

## 🚨 Fallback Mechanism

When OpenAI API is not available, the system provides:

1. **Hash-based Embeddings**: Simple but functional embeddings
2. **Rule-based Responses**: Predefined responses for common questions
3. **Context Awareness**: Still uses stored analysis data
4. **Graceful Degradation**: System continues to work

### **Fallback Response Examples**
- Model type questions → Generic model information
- Feature importance → Standard XAI explanations
- Performance questions → Basic metric descriptions

## 📈 Performance Considerations

### **Memory Usage**
- In-memory vector storage (for development)
- Consider persistent vector DB for production
- User data cleanup to manage memory

### **API Costs**
- OpenAI embedding costs: ~$0.0001 per 1K tokens
- OpenAI chat costs: ~$0.002 per 1K tokens
- Monitor usage with OpenAI dashboard

### **Response Time**
- Embedding generation: ~1-2 seconds
- Vector search: ~100-500ms
- GPT response: ~2-5 seconds
- Total: ~3-8 seconds per question

## 🔮 Future Enhancements

### **Planned Features**
1. **Persistent Vector Database**: Pinecone, Weaviate, or Chroma
2. **Advanced Caching**: Redis for response caching
3. **Multi-modal RAG**: Image understanding with CLIP
4. **Conversation Memory**: Chat history tracking
5. **Custom Prompts**: User-defined response styles

### **Production Considerations**
1. **Scalability**: Load balancing and horizontal scaling
2. **Monitoring**: Prometheus metrics and logging
3. **Backup**: Vector database persistence
4. **Security**: API key rotation and encryption

## 🐛 Troubleshooting

### **Common Issues**

#### **OpenAI API Errors**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### **Vector Database Issues**
```bash
# Check service logs
docker-compose logs ai_outputs

# Restart service
docker-compose restart ai_outputs
```

#### **Memory Issues**
```bash
# Clear user data
curl -X DELETE http://localhost:8001/clear-user-data/admin

# Check memory usage
docker stats
```

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAG Architecture Guide](https://arxiv.org/abs/2005.11401)
- [Vector Database Comparison](https://zilliz.com/comparison)
- [XAI Best Practices](https://christophm.github.io/interpretable-ml-book/)

---

**Note**: This RAG system provides a foundation for intelligent XAI conversations. For production use, consider implementing persistent vector storage and additional security measures. 