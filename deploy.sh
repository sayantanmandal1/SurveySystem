#!/bin/bash

# Ultimate AI Surveillance System Deployment Script

echo "🚀 Starting Ultimate AI Surveillance System Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models exp src/dashboard

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting the application..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access the dashboard at: http://localhost:8501"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop application: docker-compose down"
echo "  - Restart application: docker-compose restart"
echo "  - Update application: docker-compose pull && docker-compose up -d"