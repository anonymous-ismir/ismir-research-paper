# Audio Generation API Server

A FastAPI server for generating audio cues and final audio from story text.

## Features

- **API Key Authentication**: Secure API key-based authentication
- **Decide Audio Cues**: Analyze story text and extract audio cues
- **Generate Audio**: Generate audio from audio cues using specialist models
- **Complete Pipeline**: End-to-end audio generation from story text
- **CORS Support**: Cross-origin requests enabled

## Installation

1. Install dependencies:

```bash
pip install -r requirements_server.txt
```

2. Set environment variables (optional):

```bash
export API_KEYS="your-key-1,your-key-2,dev-key-123"
export PORT=8000
export HOST=0.0.0.0
```

## Running the Server

```bash
python server.py
```

Or using uvicorn directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Health Check

```
GET /api/v1/health
```

Returns server health status.

### 2. Decide Audio Cues

```
POST /api/v1/decide-cues
Headers:
  X-API-Key: your-api-key
Content-Type: application/json

Body:
{
  "story_text": "Suddenly rain started so i ran to shelter",
  "speed_wps": 2.0
}
```

Response:

```json
{
  "cues": [
    {
      "audio_class": "rain falling",
      "audio_type": "AMBIENCE",
      "start_time_ms": 10,
      "duration_ms": 2000,
      "weight_db": 0.0,
      "fade_ms": 500
    }
  ],
  "total_duration_ms": 5000,
  "message": "Successfully generated 3 audio cues"
}
```

### 3. Generate Audio from Cues

```
POST /api/v1/generate-audio
Headers:
  X-API-Key: your-api-key
Content-Type: application/json

Body:
{
  "cues": [...],
  "total_duration_ms": 5000
}
```

Response:

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "duration_ms": 5000,
  "message": "Successfully generated audio"
}
```

### 4. Generate Audio from Story (Complete Pipeline)

```
POST /api/v1/generate-from-story
Headers:
  X-API-Key: your-api-key
Content-Type: application/json

Body:
{
  "story_text": "Suddenly rain started so i ran to shelter",
  "speed_wps": 2.0,
  "return_cues": true
}
```

Response:

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "duration_ms": 5000,
  "cues": [...],
  "message": "Successfully generated audio from story with 3 cues"
}
```

### 5. Generate New API Key

```
GET /api/v1/generate-api-key
```

Returns a new API key (for admin use).

## API Key Management

### Default API Keys

The server comes with default API keys for development:

- `dev-key-123`
- `test-key-456`

### Setting Custom API Keys

Set the `API_KEYS` environment variable:

```bash
export API_KEYS="key1,key2,key3"
```

### Using API Keys

Include the API key in the request header:

```python
headers = {
    "X-API-Key": "your-api-key",
    "Content-Type": "application/json"
}
```

## Example Usage

See `api_client_example.py` for complete examples.

### Python Example

```python
import requests

url = "http://localhost:8000/api/v1/generate-from-story"
headers = {
    "X-API-Key": "dev-key-123",
    "Content-Type": "application/json"
}
data = {
    "story_text": "A dog barked loudly in the distance",
    "speed_wps": 2.0,
    "return_cues": True
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

# Decode and save audio
import base64
audio_bytes = base64.b64decode(result["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/generate-from-story" \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "story_text": "A dog barked loudly in the distance",
    "speed_wps": 2.0
  }'
```

## Response Format

### Audio Base64 Encoding

All audio responses are base64-encoded WAV files. To decode:

```python
import base64
audio_bytes = base64.b64decode(audio_base64)
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `401`: Unauthorized (invalid or missing API key)
- `422`: Validation error (invalid request body)
- `500`: Internal server error

Error response format:

```json
{
  "detail": "Error message here"
}
```

## Production Deployment

For production deployment:

1. **Secure API Keys**: Use a database or secure key management system
2. **HTTPS**: Use HTTPS with SSL certificates
3. **Rate Limiting**: Implement rate limiting per API key
4. **Logging**: Set up proper logging and monitoring
5. **CORS**: Restrict CORS origins to specific domains
6. **Admin Endpoint**: Protect the API key generation endpoint

Example with environment variables:

```bash
export API_KEYS="prod-key-1,prod-key-2"
export PORT=8000
export HOST=0.0.0.0
export LOG_LEVEL=INFO
```

## Troubleshooting

### Import Errors

Make sure all project dependencies are installed and the project structure is correct.

### API Key Issues

- Check that the `X-API-Key` header is included
- Verify the API key is in the `API_KEYS` environment variable
- Check server logs for authentication errors

### Audio Generation Errors

- Check that specialist models are properly configured
- Verify audio generation functions are working
- Check server logs for detailed error messages
