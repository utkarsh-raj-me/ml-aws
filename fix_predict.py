"""
Script to fix the 'Model Not Found' bug in main.py
This adds storage fallback to the predict endpoints
"""

import re

# Read the file
with open('app/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Update the predict endpoint
old_predict = '''@app.post("/predict")
async def predict(request: PredictRequest):
    """Make a single prediction"""
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_data = trained_models[request.model_id]'''

new_predict = '''@app.post("/predict")
async def predict(request: PredictRequest):
    """Make a single prediction"""
    # Try to get from memory first
    if request.model_id not in trained_models:
        # Try to load from storage
        model_data = storage.load_model(request.model_id)
        if model_data is None:
            raise HTTPException(status_code=404, detail="Model not found")
        # Cache in memory for future requests
        trained_models[request.model_id] = model_data
    else:
        model_data = trained_models[request.model_id]
    
    try:'''

# Fix 2: Update the batch_predict endpoint
old_batch = '''@app.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest):
    """Make batch predictions"""
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_data = trained_models[request.model_id]'''

new_batch = '''@app.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest):
    """Make batch predictions"""
    # Try to get from memory first
    if request.model_id not in trained_models:
        # Try to load from storage
        model_data = storage.load_model(request.model_id)
        if model_data is None:
            raise HTTPException(status_code=404, detail="Model not found")
        # Cache in memory for future requests
        trained_models[request.model_id] = model_data
    else:
        model_data = trained_models[request.model_id]
    
    try:'''

# Apply fixes
content = content.replace(old_predict, new_predict)
content = content.replace(old_batch, new_batch)

# Write back
with open('app/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed the predict endpoints!")
print("Models will now be loaded from storage if not in memory.")
