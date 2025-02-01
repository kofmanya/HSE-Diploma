from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from app.utils import load_components, preprocess_image, classify_image, find_similar_item, find_top_k_similar_items

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained components
model, model_name, embeddings, ids, labels, zalando_items, transform = load_components()

@app.get("/")
def home():
    return {"message": "Welcome to the Zalando Item Finder API"}

# @app.post("/search/")
# async def search_item(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess the uploaded image
#         image = Image.open(file.file).convert("RGB")
#         processed_image = preprocess_image(image, transform)

#         # Classify the image
#         category, category_label = classify_image(model, processed_image, model_name)

#         # Find the most similar item
#         similar_item = find_similar_item(processed_image, model, model_name, embeddings, ids, labels, zalando_items, category)

#         return JSONResponse(content={
#             "category": category_label,
#             "similar_item": similar_item
#         })

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/search/")
async def search_item(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image, transform)

        # Classify the image to get category
        category, category_label = classify_image(model, processed_image, model_name)

        # Find the top 3 most similar items in that category
        similar_items = find_top_k_similar_items(processed_image, model, model_name, embeddings, ids, labels,zalando_items, category=category, k=3)

        return JSONResponse(content={
            "category": category_label,
            "similar_items": similar_items
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
