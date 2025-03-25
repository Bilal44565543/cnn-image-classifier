from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
from src.models.cnn import CNNModel
from src.data.preprocessor import DataPreprocessor
from src.config import DEVICE_CONFIG, MODEL_CONFIG, PATHS

app = FastAPI(title="Image Classification API")

# Initialize model and preprocessor
device = torch.device(DEVICE_CONFIG['device'])
model = CNNModel()
model.load_state_dict(torch.load(PATHS['model_dir'] + '/' + 'best_model.pth')['model_state_dict'])
model.to(device)
model.eval()

preprocessor = DataPreprocessor(augment=False)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = preprocessor.preprocess_image(image)
        image = image.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class name
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        predicted_class_name = class_names[predicted_class]
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = [
            {
                "class": class_names[idx],
                "probability": float(prob)
            }
            for prob, idx in zip(top3_prob[0], top3_indices[0])
        ]
        
        return JSONResponse({
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "top_3_predictions": top3_predictions
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 