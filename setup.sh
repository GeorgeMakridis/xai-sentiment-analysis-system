#!/bin/bash

# XAI Dashboard Setup Script

echo "🚀 XAI Dashboard System Setup"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "   - Linux: https://docs.docker.com/engine/install/"
    echo "   - Windows: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Check available ports
echo "🔍 Checking port availability..."
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 3000 is already in use. Please free up port 3000."
    exit 1
fi

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use. Please free up port 8000."
    exit 1
fi

if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8001 is already in use. Please free up port 8001."
    exit 1
fi

echo "✅ All required ports are available"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p shared_volume/uploads
mkdir -p shared_volume/models
mkdir -p shared_volume/results

# Build and start containers
echo "🔨 Building and starting containers..."
docker compose up --build -d

# Wait for containers to start
echo "⏳ Waiting for containers to start..."
sleep 10

# Check container status
echo "📊 Container status:"
docker compose ps

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo "🌐 Dashboard: http://localhost:3000"
echo "🔑 Login credentials:"
echo "   Username: admin"
echo "   Password: password123"
echo ""
echo "📋 Next steps:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Login with the credentials above"
echo "3. Upload your data file (or use demo mode)"
echo "4. Upload your model file (or use demo mode)"
echo "5. View XAI visualizations and ask questions!"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "🛑 To stop the system: docker compose down"
echo "🔄 To restart: docker compose restart" 