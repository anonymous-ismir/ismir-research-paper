# Docker Setup Guide

This guide explains how to build and run the Cinemaudio backend using Docker.

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 2.0+)

## Quick Start

### 1. Create Environment File

Create a `.env` file in the project root or `backend/` directory:

```bash
# Server Configuration
PORT=8000
HOST=0.0.0.0

# Google Gemini API Key (required for audio cue decision)
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Build and Run (Development)

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f backend
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/docs (should return 200)

### 4. Stop the Container

```bash
docker-compose down
```

## Production Deployment

For production, use the production compose file:

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

**Note**: The production configuration doesn't mount the code as a volume, so changes require rebuilding the image.

## Building the Image Manually

```bash
cd backend
docker build -t cinemaudio-backend .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  cinemaudio-backend
```

## GPU Support (Optional)

If you have an NVIDIA GPU:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Rebuild and run:

```bash
docker-compose up --build
```

## Troubleshooting

### Model Download Time

On first run, the container will download model weights (TangoFlux, CLAP) which can take several minutes. This is normal.

### Port Already in Use

If port 8000 is already in use, change it in `.env`:

```bash
BACKEND_PORT=8001
```

### Environment Variables Not Loading

Make sure your `.env` file is in the project root or backend directory, and that docker-compose can access it.

### Build Fails on TangoFlux Installation

TangoFlux is installed from GitHub. Make sure:
- Git is available in the container (already included in Dockerfile)
- You have internet connectivity
- The GitHub repository is accessible

## File Structure

```
.
├── docker-compose.yml          # Development compose file
├── docker-compose.prod.yml     # Production compose file
├── backend/
│   ├── Dockerfile              # Backend Docker image definition
│   ├── requirements.txt        # Python dependencies
│   ├── .dockerignore           # Files to exclude from build
│   └── DOCKER_README.md         # Detailed Docker documentation
└── .env                        # Environment variables (create this)
```

## Additional Notes

- Model cache is persisted in a Docker volume (`model_cache`)
- For development, code is mounted as a volume for hot-reload
- For production, code is baked into the image
- Health checks are configured to monitor container status

