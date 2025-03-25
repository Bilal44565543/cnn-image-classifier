# API Documentation

## Overview

The CNN Image Classifier provides a RESTful API built with FastAPI for image classification. The API allows users to upload images and receive predictions with confidence scores.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Welcome Endpoint

```http
GET /
```

Returns a welcome message and basic information about the API.

#### Response

```json
{
    "message": "Welcome to the CNN Image Classification API",
    "version": "1.0.0",
    "endpoints": {
        "predict": "/predict"
    }
}
```

### 2. Prediction Endpoint

```http
POST /predict
```

Classifies an uploaded image and returns the predicted class with confidence scores.

#### Request

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (supported formats: JPG, PNG)

#### Response

```json
{
    "class": "airplane",
    "confidence": 0.95,
    "top_3_predictions": [
        {
            "class": "airplane",
            "confidence": 0.95
        },
        {
            "class": "bird",
            "confidence": 0.03
        },
        {
            "class": "cat",
            "confidence": 0.02
        }
    ]
}
```

#### Error Responses

1. No File Uploaded
```json
{
    "error": "No file uploaded"
}
```

2. Invalid File Type
```json
{
    "error": "Invalid file type. Supported formats: JPG, PNG"
}
```

3. Processing Error
```json
{
    "error": "Error processing image"
}
```

## Example Usage

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## Rate Limiting

Currently, there are no rate limits implemented. However, please be mindful of server resources when making requests.

## CORS

The API supports CORS and can be accessed from any origin. For production deployment, you may want to restrict this to specific domains.

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 415: Unsupported Media Type
- 500: Internal Server Error

## Security

- File size limit: 10MB
- Supported file types: JPG, PNG
- Input validation for all requests
- Error messages are sanitized to prevent information leakage

## Future Enhancements

1. Authentication
2. Rate limiting
3. Batch processing
4. Additional model endpoints
5. Model versioning 