import torch
import numpy as np
import pickle
import json
import os
from torchvision import transforms, models
import torch.nn as nn
from transformers import ViTForImageClassification
import cv2
import timm

# Category mapping
CATEGORY_LABELS = {
    0: "Coats",
    1: "Dresses",
    2: "Jackets & Blazers",
    3: "Jeans",
    4: "Shirts & Blouses",
    5: "T-shirts & Tops"
}

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
    model_folder = "models"
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]

    if not model_files:
        raise ValueError("No model file found in the 'models' folder.")

    if len(model_files) > 1:
        raise ValueError(f"Multiple model files found: {model_files}. Please keep only one.")

    model_file = model_files[0]
    model_name = os.path.splitext(model_file)[0].lower()

    if model_name.lower() == "resnet34":  
        # 1. Define the model architecture (ResNet34)
        model = models.resnet34(weights=None)
        num_classes = 6
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, num_classes)
        )
    elif model_name.lower() == "vgg16": 
        # 2. Define the model architecture (VGG16)
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_classes = 6
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_name.lower() == "dinov2": 
        # 3. Define the model architecture (DINOv2)
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in backbone.parameters():
            param.requires_grad = False  # Freeze backbone parameters

        class DINOv2Classifier(nn.Module):
            def __init__(self, backbone, hidden_size, num_classes):
                super(DINOv2Classifier, self).__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                # Pass through the backbone
                embeddings = self.backbone(x)
                # Pass through the classification head
                logits = self.classifier(embeddings)
                return logits

        num_classes = 6
        hidden_size = 384  
        model = DINOv2Classifier(backbone=backbone, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name.lower() == "tnt":
        # 4. Define the model architecture (TNT)
        pretrained_model_name = "tnt_s_patch16_224"
        model = timm.create_model(pretrained_model_name, pretrained=False)

        # Modify the classifier to fit the number of classes
        num_classes = 6
        hidden_features = model.head.in_features
        model.head = torch.nn.Linear(hidden_features, num_classes)
    elif model_name.lower() == "vit":
        # 5. Define the model architecture (ViT)
        pretrained_model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(pretrained_model_name, num_labels=6)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

     # Load trained weights
    model_path = f"models/{model_name}.pth"  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  

    # Load corresponding embeddings and transformations
    data_folder = "data"
    embeddings_file = os.path.join(data_folder, f"{model_name}_embeddings.npz")
    transformations_file = os.path.join(data_folder, f"{model_name}_transformations.json")

    if not os.path.exists(embeddings_file):
        raise ValueError(f"Embeddings file '{embeddings_file}' not found.")
    if not os.path.exists(transformations_file):
        raise ValueError(f"Transformations file '{transformations_file}' not found.")

    # Load corresponding embeddings
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    embeddings = embeddings_data["embeddings"]
    labels = embeddings_data["labels"]
    ids = embeddings_data["ids"]

    # Load Zalando items
    with open("data/items.pkl", "rb") as f:
        zalando_items = pickle.load(f)

    # Load corresponding transformations
    transform = load_transformations(transformations_file)

    return model, model_name, embeddings, ids, labels, zalando_items, transform

def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

def classify_image(model, image, model_name):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.logits if model_name.lower() == "vit" else output, 1)
    return predicted.item(), CATEGORY_LABELS[predicted.item()]

def find_similar_item(image, model, model_name, embeddings, ids, labels, zalando_items, category=None):
    with torch.no_grad():
        # Compute the image embedding
        output = model(image)

        image_embedding = (output.logits if model_name.lower() == "vit" else output).cpu().numpy().flatten()

        # Filter embeddings by category if specified
        if category:
            filtered_indices = [i for i, label in enumerate(labels) if label == category]
            if not filtered_indices:
                raise ValueError(f"No embeddings found for category '{category}'.")
            
            filtered_embeddings = embeddings[filtered_indices]
            filtered_ids = [ids[i] for i in filtered_indices]
        else:
            # Use all embeddings if no category is specified
            filtered_embeddings = embeddings
            filtered_ids = ids

        # Compute distances
        distances = np.linalg.norm(filtered_embeddings - image_embedding, axis=1)

        # Sort all distances ascending
        sorted_indices = np.argsort(distances)

        # Iterate over candidates from nearest to farthest
        for idx in sorted_indices:
            candidate_id = filtered_ids[idx]
            candidate_basename = candidate_id.split("/")[-1]  # e.g. 'ef0e2a17b8a44471.jpg'

            # Try to find a matching Zalando item
            candidate_item = next(
                (item for item in zalando_items.values()
                 if any(candidate_basename in img_url for img_url in item["images"])),
                None
            )

            # If found, return it
            if candidate_item:
                candidate_item["images"] = candidate_item["images"][:5]
                return candidate_item

        # If we've exhausted all candidates without finding a match:
        raise ValueError("No matching Zalando item found for any candidate image ID.")

def find_top_k_similar_items(image, model, model_name, embeddings, ids, labels, zalando_items, category=None, k=3):
    with torch.no_grad():
        # Compute the image embedding
        output = model(image)
        image_embedding = (output.logits if model_name.lower() == "vit" else output).cpu().numpy().flatten()

        # Filter embeddings by category if specified
        if category is not None:
            filtered_indices = [i for i, label in enumerate(labels) if label == category]
            if not filtered_indices:
                raise ValueError(f"No embeddings found for category '{category}'.")
            
            filtered_embeddings = embeddings[filtered_indices]
            filtered_ids = [ids[i] for i in filtered_indices]
        else:
            filtered_embeddings = embeddings
            filtered_ids = ids

        # Compute distances
        distances = np.linalg.norm(filtered_embeddings - image_embedding, axis=1)

        # Sort all distances ascending
        sorted_indices = np.argsort(distances)

        # Collect up to k valid Zalando items
        similar_items = []
        for idx in sorted_indices:
            candidate_id = filtered_ids[idx]
            candidate_basename = candidate_id.split("/")[-1]  # e.g. 'ef0e2a17b8a44471.jpg'

            # Try to find a matching Zalando item from your items dictionary
            candidate_item = next(
                (item for item in zalando_items.values()
                 if any(candidate_basename in img_url for img_url in item["images"])),
                None
            )

            if candidate_item:
                # Optionally limit the images to the first 5 for display
                candidate_item["images"] = candidate_item["images"][:5]
                similar_items.append(candidate_item)

            if len(similar_items) == k:
                break

        if not similar_items:
            raise ValueError("No matching Zalando items found.")
        
        return similar_items
