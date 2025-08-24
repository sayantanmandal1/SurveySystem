@echo off
REM Ultimate AI Surveillance System Deployment Script for Windows

echo 🚀 Starting Ultimate AI Surveillance System Deployment...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "exp" mkdir exp
if not exist "src\dashboard" mkdir src\dashboard

REM Build and start the application
echo 🔨 Building Docker image...
docker-compose build

echo 🚀 Starting the application...
docker-compose up -d

REM Wait for the application to start
echo ⏳ Waiting for application to start...
timeout /t 30 /nobreak >nul

REM Check if the application is running
curl -f http://localhost:8501/_stcore/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Application is running successfully!
    echo 🌐 Access the dashboard at: http://localhost:8501
) else (
    echo ❌ Application failed to start. Check logs with: docker-compose logs
    pause
    exit /b 1
)

echo 🎉 Deployment completed successfully!
echo.
echo 📋 Useful commands:
echo   - View logs: docker-compose logs -f
echo   - Stop application: docker-compose down
echo   - Restart application: docker-compose restart
echo   - Update application: docker-compose pull ^&^& docker-compose up -d

pause