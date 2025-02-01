#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
from app.utils import load_components, preprocess_image, classify_image, find_similar_item

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained components
model, embeddings, zalando_items, transform = load_components()

@app.get("/")
def root():
    return {"message": "Welcome to the Zalando Item Finder API"}

@app.post("/search/")
async def search_item(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image, transform)

        # Classify the image
        category = classify_image(model, processed_image)

        # Find the most similar item
        similar_item = find_similar_item(processed_image, model, embeddings, zalando_items)

        return JSONResponse(content={
            "category": category,
            "similar_item": similar_item
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

