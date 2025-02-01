#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import pickle
import json
from torchvision import models, transforms

def load_transformations(transformations_file):
    with open(transformations_file, "r") as f:
        transformations = json.load(f)

    transform_list = []
    for transform in transformations:
        name = transform["name"]
        params = transform["params"]

        if name == "Resize":
            transform_list.append(transforms.Resize(tuple(params["size"])))
        elif name == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip(p=params["p"]))
        elif name == "RandomRotation":
            transform_list.append(transforms.RandomRotation(degrees=tuple(params["degrees"])))
        elif name == "ColorJitter":
            transform_list.append(transforms.ColorJitter(
                brightness=params["brightness"],
                contrast=params["contrast"],
                saturation=params["saturation"],
                hue=params["hue"]
            ))
        elif name == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif name == "Normalize":
            transform_list.append(transforms.Normalize(mean=params["mean"], std=params["std"]))
        else:
            raise ValueError(f"Unknown transformation: {name}")

    return transforms.Compose(transform_list)

def load_components():
    # Define the model architecture (ResNet34)
    model = models.resnet34(weights=None)
    num_classes = 6
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, num_classes)
    )
    model_path = "app/models/model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load embeddings
    embeddings_data = np.load("app/data/model_embeddings.npz")
    embeddings = embeddings_data["embeddings"]

    # Load Zalando items
    with open("app/data/downloaded_items.pkl", "rb") as f:
        zalando_items = pickle.load(f)

    # Load transformations
    transform = load_transformations("app/data/transformations.json")

    return model, embeddings, zalando_items, transform

# Preprocess image
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

# Classify image
def classify_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Find the most similar item
def find_similar_item(image, model, embeddings, zalando_items):
    with torch.no_grad():
        # Pass the image through the model to compute the embedding
        output = model(image)  # Output shape: (1, 6)
        image_embedding = output.cpu().numpy().flatten()  # Shape: (6,)

        # Compute similarity (e.g., cosine similarity)
        similarities = np.dot(embeddings, image_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(image_embedding)
        )

        # Find the most similar item
        most_similar_index = np.argmax(similarities)
        return zalando_items[most_similar_index]

