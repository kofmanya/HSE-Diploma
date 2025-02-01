import torch
import numpy as np
import pickle
import json
import os
import boto3
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

# AWS S3 Configuration
S3_BUCKET_NAME = "hse-diploma-models" 
S3_REGION = "us-east-1"  
S3_MODEL_DIR = "models/"
S3_DATA_DIR = "data/"

# Local directories
LOCAL_MODEL_DIR = "models"
LOCAL_DATA_DIR = "data"

# Local file path for items.pkl (this is stored locally, NOT in S3)
LOCAL_ITEMS_FILE = os.path.join(LOCAL_DATA_DIR, "items.pkl")

# Ensure local directories exist
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Initialize S3 client
s3 = boto3.client("s3")

def download_from_s3(s3_key, local_path):
    """Downloads a file from S3 to a local directory."""
    if not os.path.exists(local_path):  
        print(f"Downloading {s3_key} from S3...")
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        print(f"Saved to {local_path}")
    else:
        print(f"{local_path} already exists, skipping download.")

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

# def load_components():
#     model_folder = "models"
#     model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]

#     if not model_files:
#         raise ValueError("No model file found in the 'models' folder.")

#     if len(model_files) > 1:
#         raise ValueError(f"Multiple model files found: {model_files}. Please keep only one.")

#     model_file = model_files[0]
#     model_name = os.path.splitext(model_file)[0].lower()

#     if model_name.lower() == "resnet34":  
#         # 1. Define the model architecture (ResNet34)
#         model = models.resnet34(weights=None)
#         num_classes = 6
#         num_features = model.fc.in_features
#         model.fc = torch.nn.Sequential(
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(num_features, num_classes)
#         )
#     elif model_name.lower() == "vgg16": 
#         # 2. Define the model architecture (VGG16)
#         model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         num_classes = 6
#         num_features = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(num_features, num_classes)
#     elif model_name.lower() == "dinov2": 
#         # 3. Define the model architecture (DINOv2)
#         backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#         for param in backbone.parameters():
#             param.requires_grad = False  # Freeze backbone parameters

#         class DINOv2Classifier(nn.Module):
#             def __init__(self, backbone, hidden_size, num_classes):
#                 super(DINOv2Classifier, self).__init__()
#                 self.backbone = backbone
#                 self.classifier = nn.Linear(hidden_size, num_classes)

#             def forward(self, x):
#                 # Pass through the backbone
#                 embeddings = self.backbone(x)
#                 # Pass through the classification head
#                 logits = self.classifier(embeddings)
#                 return logits

#         num_classes = 6
#         hidden_size = 384  
#         model = DINOv2Classifier(backbone=backbone, hidden_size=hidden_size, num_classes=num_classes)
#     elif model_name.lower() == "tnt":
#         # 4. Define the model architecture (TNT)
#         pretrained_model_name = "tnt_s_patch16_224"
#         model = timm.create_model(pretrained_model_name, pretrained=False)

#         # Modify the classifier to fit the number of classes
#         num_classes = 6
#         hidden_features = model.head.in_features
#         model.head = torch.nn.Linear(hidden_features, num_classes)
#     elif model_name.lower() == "vit":
#         # 5. Define the model architecture (ViT)
#         pretrained_model_name = "google/vit-base-patch16-224-in21k"
#         model = ViTForImageClassification.from_pretrained(pretrained_model_name, num_labels=6)
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

#     # Move model to the appropriate device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#      # Load trained weights
#     model_path = f"models/{model_name}.pth"  
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()  

#     # Load corresponding embeddings and transformations
#     data_folder = "data"
#     embeddings_file = os.path.join(data_folder, f"{model_name}_embeddings.npz")
#     transformations_file = os.path.join(data_folder, f"{model_name}_transformations.json")

#     if not os.path.exists(embeddings_file):
#         raise ValueError(f"Embeddings file '{embeddings_file}' not found.")
#     if not os.path.exists(transformations_file):
#         raise ValueError(f"Transformations file '{transformations_file}' not found.")

#     # Load corresponding embeddings
#     embeddings_data = np.load(embeddings_file, allow_pickle=True)
#     embeddings = embeddings_data["embeddings"]
#     labels = embeddings_data["labels"]
#     ids = embeddings_data["ids"]

#     # Load Zalando items
#     with open("data/items.pkl", "rb") as f:
#         zalando_items = pickle.load(f)

#     # Load corresponding transformations
#     transform = load_transformations(transformations_file)

#     return model, model_name, embeddings, ids, labels, zalando_items, transform

def load_components():
    # Get model file name dynamically from S3
    model_files = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_MODEL_DIR).get("Contents", [])
    model_files = [obj["Key"] for obj in model_files if obj["Key"].endswith(".pth")]

    if not model_files:
        raise ValueError("No model file found in S3 bucket.")

    if len(model_files) > 1:
        raise ValueError(f"Multiple model files found in S3: {model_files}. Please keep only one.")

    model_s3_path = model_files[0]  # Example: "models/vit.pth"
    model_name = os.path.splitext(os.path.basename(model_s3_path))[0].lower()
    model_local_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(model_s3_path))

    # Download model file from S3
    download_from_s3(model_s3_path, model_local_path)

    # Load the appropriate model architecture
    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 6)
        )
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 6)
    elif model_name == "dinov2":
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in backbone.parameters():
            param.requires_grad = False  # Freeze backbone

        class DINOv2Classifier(nn.Module):
            def __init__(self, backbone, hidden_size, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                embeddings = self.backbone(x)
                return self.classifier(embeddings)

        model = DINOv2Classifier(backbone, hidden_size=384, num_classes=6)
    elif model_name == "tnt":
        model = timm.create_model("tnt_s_patch16_224", pretrained=False)
        num_features = model.head.in_features
        model.head = torch.nn.Linear(num_features, 6)
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=6)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_local_path, map_location=device))
    model.eval()
    model.to(device)

    # Download embeddings & transformations from S3
    embeddings_s3_path = f"{S3_DATA_DIR}{model_name}_embeddings.npz"
    transformations_s3_path = f"{S3_DATA_DIR}{model_name}_transformations.json"

    embeddings_local_path = os.path.join(LOCAL_DATA_DIR, f"{model_name}_embeddings.npz")
    transformations_local_path = os.path.join(LOCAL_DATA_DIR, f"{model_name}_transformations.json")

    download_from_s3(embeddings_s3_path, embeddings_local_path)
    download_from_s3(transformations_s3_path, transformations_local_path)

    # Load embeddings
    embeddings_data = np.load(embeddings_local_path, allow_pickle=True)
    embeddings = embeddings_data["embeddings"]
    labels = embeddings_data["labels"]
    ids = embeddings_data["ids"]

    # Load Zalando items (LOCALLY, not from S3)
    if not os.path.exists(LOCAL_ITEMS_FILE):
        raise ValueError(f"Local items file '{LOCAL_ITEMS_FILE}' not found. Please ensure it is placed in {LOCAL_DATA_DIR}.")
    
    with open(LOCAL_ITEMS_FILE, "rb") as f:
        zalando_items = pickle.load(f)

    # Load transformations
    with open(transformations_local_path, "r") as f:
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

    transform = transforms.Compose(transform_list)

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
