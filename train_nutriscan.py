"""
NutriScanAI: Comprehensive Training Script
==========================================
This script trains all models for the NutriScanAI system:
1. MobileNetV2-based CNN for 6-class vitamin deficiency classification
2. Custom MLP for cholesterol prediction
3. Fine-tuned BLIP model for medical image captioning
4. XAI integration (LIME, Grad-CAM, Integrated Gradients, SHAP)

Optimized for Google Colab with GPU support.

Requirements:
- Install: pip install torch torchvision tqdm matplotlib seaborn scikit-learn 
           opencv-python pillow numpy pandas lime shap transformers

Usage:
    python train_nutriscan.py

For Google Colab:
    !pip install torch torchvision tqdm matplotlib seaborn scikit-learn 
    opencv-python pillow numpy pandas lime shap transformers
    !python train_nutriscan.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize, StandardScaler
import cv2
import pandas as pd
from PIL import Image
import logging
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class tqdm:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.n = 0
        def __iter__(self):
            if self.desc:
                print(self.desc)
            return iter(self.iterable)
        def __next__(self):
            return next(iter(self.iterable))
        def set_postfix(self, **kwargs):
            pass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# XAI imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

# Transformers for BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
    from transformers import Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

device = get_device()

# ============================================================================
# 1. MOBILENETV2 CNN MODEL FOR VITAMIN DEFICIENCY CLASSIFICATION
# ============================================================================

class VitaminDataset(Dataset):
    """Dataset class for vitamin deficiency images."""
    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if classes is None:
            self.classes = sorted([d for d in os.listdir(root_dir) 
                                  if os.path.isdir(os.path.join(root_dir, d))])
        else:
            self.classes = classes
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                    # Skip mask directories
                    if 'mask' in img_name.lower() or 'mask' in class_dir.lower():
                        continue
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        for cls, idx in self.class_to_idx.items():
            count = self.labels.count(idx)
            logger.info(f"  {cls}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                placeholder = self.transform(placeholder)
            return placeholder, label

def get_data_transforms():
    """Get data transforms with augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_mobilenet_model(num_classes=6):
    """Create MobileNetV2 model for classification."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify classifier for our number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model

def train_cnn_model(
    dataset_dir="dataset",
    epochs=65,
    batch_size=32,
    learning_rate=0.001,
    patience=3,
    scheduler_factor=0.5,
    save_path="mobilenet_vitamin.pth",
    classes=None
):
    """
    Train MobileNetV2 CNN model for vitamin deficiency classification.
    
    Parameters:
    - epochs: 65 (as specified)
    - batch_size: 32 (as specified)
    - learning_rate: 0.001 (as specified)
    - patience: 3 (for ReduceLROnPlateau, as specified)
    - scheduler_factor: 0.5 (as specified)
    """
    logger.info("=" * 80)
    logger.info("TRAINING MOBILENETV2 CNN MODEL")
    logger.info("=" * 80)
    
    # Default classes
    if classes is None:
        classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
    
    num_classes = len(classes)
    logger.info(f"Classes: {classes}")
    logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    full_dataset = VitaminDataset(dataset_dir, transform=train_transform, classes=classes)
    
    if len(full_dataset) == 0:
        logger.error("No images found in dataset directory!")
        return None, [], [], [], []
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update transforms for validation
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = create_mobilenet_model(num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=scheduler_factor, verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ Best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping (optional, not in original spec but good practice)
        if patience_counter >= 10:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    # Test evaluation
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL TEST ACCURACY: {test_accuracy:.2f}%")
    logger.info(f"{'='*80}\n")
    
    # Generate evaluation metrics
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}\n")
    
    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(all_labels, all_predictions, target_names=classes)}\n")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies, {
        'test_accuracy': test_accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classes': classes
    }

# ============================================================================
# 2. MLP MODEL FOR CHOLESTEROL PREDICTION
# ============================================================================

class CholesterolMLP(nn.Module):
    """MLP model for cholesterol prediction."""
    def __init__(self, num_features, num_classes=2, dropout1=0.3, dropout2=0.2):
        super(CholesterolMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_mlp_model(
    csv_path="dataset/dataset_2190_cholesterol.csv",
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    patience=3,
    save_path="cholesterol_mlp.pth"
):
    """
    Train MLP model for cholesterol prediction.
    
    Parameters:
    - epochs: 20 (as specified)
    - batch_size: 32 (as specified)
    - dropout rates: 0.3, 0.2 (as specified)
    """
    logger.info("=" * 80)
    logger.info("TRAINING MLP MODEL FOR CHOLESTEROL PREDICTION")
    logger.info("=" * 80)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        logger.warning(f"Cholesterol CSV not found at {csv_path}. Skipping MLP training.")
        logger.info("To train MLP model, please provide a CSV file with features and target column.")
        return None, [], [], [], []
    
    try:
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with shape: {df.shape}")
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Check for target column
        if 'target' in df.columns:
            target_column = 'target'
        elif 'num' in df.columns:
            target_column = 'num'
        else:
            logger.error("No target column found. Expected 'target' or 'num'.")
            return None, [], [], [], []
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != target_column]
        features = df[feature_columns]
        labels = df[target_column]
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_df = pd.DataFrame(features_scaled, columns=feature_columns)
        
        # Determine number of classes
        num_classes = len(labels.unique())
        num_features = features_df.shape[1]
        logger.info(f"Features: {num_features}, Classes: {num_classes}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_df, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test.values, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = CholesterolMLP(num_features, num_classes, dropout1=0.3, dropout2=0.2)
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=0.5, verbose=True
        )
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                logger.info(f"✓ Best model saved!")
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL TEST ACCURACY: {test_accuracy:.2f}%")
        logger.info(f"{'='*80}\n")
        
        return model, train_losses, val_losses, train_accuracies, val_accuracies
        
    except Exception as e:
        logger.error(f"Error training MLP model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, [], [], [], []

# ============================================================================
# 3. BLIP MODEL FINE-TUNING FOR MEDICAL IMAGE CAPTIONING
# ============================================================================

def fine_tune_blip_model(
    dataset_dir="dataset",
    epochs=5,
    batch_size=8,
    learning_rate=5e-5,
    max_length=100,
    num_beams=5,
    temperature=0.7,
    save_path="blip_medical_captioning"
):
    """
    Fine-tune BLIP model for medical image captioning.
    
    Parameters:
    - max_length: 100 (as specified)
    - num_beams: 5 (as specified)
    - temperature: 0.7 (as specified)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available. Skipping BLIP fine-tuning.")
        return None
    
    logger.info("=" * 80)
    logger.info("FINE-TUNING BLIP MODEL FOR MEDICAL IMAGE CAPTIONING")
    logger.info("=" * 80)
    
    try:
        # Load pre-trained BLIP model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = model.to(device)
        
        logger.info("BLIP model loaded successfully")
        
        # Prepare dataset (simplified - in practice, you'd need proper medical captions)
        # For now, we'll use generic captions based on class names
        classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        
        # Create a simple dataset with image-caption pairs
        dataset_items = []
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Generate medical captions
            medical_captions = {
                "Vitamin A": "Medical image showing potential vitamin A deficiency indicators",
                "Vitamin B": "Medical image showing potential vitamin B deficiency indicators",
                "Vitamin C": "Medical image showing potential vitamin C deficiency indicators",
                "Vitamin D": "Medical image showing potential vitamin D deficiency indicators",
                "Vitamin E": "Medical image showing potential vitamin E deficiency indicators",
                "Retina Blood Vessel": "Retinal fundus image showing blood vessel patterns for medical analysis"
            }
            
            caption = medical_captions.get(class_name, f"Medical image for {class_name} analysis")
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                    if 'mask' in img_name.lower():
                        continue
                    img_path = os.path.join(class_dir, img_name)
                    dataset_items.append((img_path, caption))
        
        if len(dataset_items) == 0:
            logger.warning("No images found for BLIP fine-tuning. Skipping.")
            return None
        
        logger.info(f"Prepared {len(dataset_items)} image-caption pairs")
        
        # Simple fine-tuning loop (simplified version)
        # In production, use proper HuggingFace Trainer
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Sample a subset for fine-tuning (to save time)
        sample_size = min(100, len(dataset_items))
        dataset_items = dataset_items[:sample_size]
        
        logger.info(f"Fine-tuning on {len(dataset_items)} samples for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (img_path, caption) in enumerate(tqdm(dataset_items, desc=f"Epoch {epoch+1}/{epochs}")):
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                    # Prepare inputs
                    inputs = processor(images=image, text=caption, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(dataset_items)
            logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save model
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        logger.info(f"✓ BLIP model saved to {save_path}")
        
        # Test generation
        model.eval()
        test_image_path = dataset_items[0][0]
        test_image = Image.open(test_image_path).convert('RGB')
        inputs = processor(images=test_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature
            )
        
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        logger.info(f"Sample caption: {generated_text}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error fine-tuning BLIP model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# 4. XAI (EXPLAINABLE AI) FUNCTIONS
# ============================================================================

def apply_lime_explanation(model, image, classes, num_samples=1000):
    """Apply LIME explanation to image."""
    if not LIME_AVAILABLE:
        logger.warning("LIME not available. Skipping LIME explanation.")
        return None
    
    try:
        explainer = LimeImageExplainer()
        
        def predict_fn(images):
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            batch = torch.stack([transform(Image.fromarray(img)) for img in images]).to(device)
            with torch.no_grad():
                outputs = model(batch)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
        explanation = explainer.explain_instance(
            np.array(image),
            predict_fn,
            top_labels=len(classes),
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error applying LIME: {e}")
        return None

def apply_gradcam(model, image, target_class, layer_name=None):
    """Apply Grad-CAM explanation."""
    try:
        # Convert PIL to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        # Find the last convolutional layer
        if layer_name is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    layer_name = name
                    target_layer = module
        
        if layer_name is None:
            logger.warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Register hooks
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = model(input_tensor)
        target = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target.backward()
        
        # Calculate Grad-CAM
        if activations and gradients:
            weights = torch.mean(gradients[0], dim=[2, 3])
            cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations[0], dim=1)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            handle_forward.remove()
            handle_backward.remove()
            
            return cam
        
        handle_forward.remove()
        handle_backward.remove()
        return None
        
    except Exception as e:
        logger.error(f"Error applying Grad-CAM: {e}")
        return None

def apply_integrated_gradients(model, image, target_class, steps=50):
    """Apply Integrated Gradients explanation."""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        baseline = torch.zeros_like(input_tensor)
        
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            output = model(interpolated)
            target = output[0, target_class]
            
            model.zero_grad()
            target.backward()
            
            integrated_gradients += interpolated.grad / steps
        
        # Create attribution map
        ig_map = integrated_gradients.squeeze().abs().mean(dim=0).cpu().numpy()
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
        
        return ig_map
        
    except Exception as e:
        logger.error(f"Error applying Integrated Gradients: {e}")
        return None

def apply_shap_explanation(model, image, classes, num_samples=50):
    """Apply SHAP explanation (simplified version)."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping SHAP explanation.")
        return None
    
    try:
        # SHAP for images is computationally expensive
        # This is a simplified version
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        def model_wrapper(x):
            # x is a batch of images
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
        # Create a masker (simplified)
        masker = shap.maskers.Image("inpaint_telea", input_tensor.shape[1:])
        
        # Create explainer
        explainer = shap.Explainer(model_wrapper, masker)
        
        # Explain
        shap_values = explainer(input_tensor, max_evals=num_samples, batch_size=1)
        
        return shap_values
        
    except Exception as e:
        logger.error(f"Error applying SHAP: {e}")
        return None

def visualize_xai_results(image, lime_exp, gradcam_map, ig_map, classes, save_path="xai_results.png"):
    """Visualize all XAI results in a single figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # LIME
    if lime_exp is not None:
        try:
            temp, mask = lime_exp.get_image_and_mask(
                lime_exp.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=True
            )
            axes[0, 1].imshow(mask, cmap='hot')
            axes[0, 1].set_title('LIME Explanation', fontsize=14, fontweight='bold')
        except:
            axes[0, 1].text(0.5, 0.5, 'LIME not available', ha='center', va='center')
    else:
        axes[0, 1].text(0.5, 0.5, 'LIME not available', ha='center', va='center')
    axes[0, 1].axis('off')
    
    # Grad-CAM
    if gradcam_map is not None:
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(gradcam_map, alpha=0.6, cmap='jet')
        axes[1, 0].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Grad-CAM not available', ha='center', va='center')
    axes[1, 0].axis('off')
    
    # Integrated Gradients
    if ig_map is not None:
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(ig_map, alpha=0.6, cmap='hot')
        axes[1, 1].set_title('Integrated Gradients', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Integrated Gradients not available', ha='center', va='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"XAI visualization saved to {save_path}")

# ============================================================================
# 5. MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function that orchestrates all model training."""
    logger.info("=" * 80)
    logger.info("NUTRISCANAI COMPREHENSIVE TRAINING SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Configuration
    config = {
        'cnn': {
            'dataset_dir': 'dataset',
            'epochs': 65,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 3,
            'scheduler_factor': 0.5,
            'save_path': 'mobilenet_vitamin.pth',
            'classes': ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        },
        'mlp': {
            'csv_path': 'dataset/dataset_2190_cholesterol.csv',
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 3,
            'save_path': 'cholesterol_mlp.pth'
        },
        'blip': {
            'dataset_dir': 'dataset',
            'epochs': 5,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'max_length': 100,
            'num_beams': 5,
            'temperature': 0.7,
            'save_path': 'blip_medical_captioning'
        }
    }
    
    results = {}
    
    # 1. Train CNN Model
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Training MobileNetV2 CNN Model")
    logger.info("="*80 + "\n")
    
    cnn_model, cnn_train_losses, cnn_val_losses, cnn_train_accs, cnn_val_accs, cnn_eval = train_cnn_model(
        dataset_dir=config['cnn']['dataset_dir'],
        epochs=config['cnn']['epochs'],
        batch_size=config['cnn']['batch_size'],
        learning_rate=config['cnn']['learning_rate'],
        patience=config['cnn']['patience'],
        scheduler_factor=config['cnn']['scheduler_factor'],
        save_path=config['cnn']['save_path'],
        classes=config['cnn']['classes']
    )
    
    if cnn_model is not None:
        results['cnn'] = {
            'model': cnn_model,
            'train_losses': cnn_train_losses,
            'val_losses': cnn_val_losses,
            'train_accuracies': cnn_train_accs,
            'val_accuracies': cnn_val_accs,
            'evaluation': cnn_eval
        }
        logger.info("✓ CNN model training completed successfully!")
    else:
        logger.error("✗ CNN model training failed!")
    
    # 2. Train MLP Model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Training MLP Model for Cholesterol Prediction")
    logger.info("="*80 + "\n")
    
    mlp_model, mlp_train_losses, mlp_val_losses, mlp_train_accs, mlp_val_accs = train_mlp_model(
        csv_path=config['mlp']['csv_path'],
        epochs=config['mlp']['epochs'],
        batch_size=config['mlp']['batch_size'],
        learning_rate=config['mlp']['learning_rate'],
        patience=config['mlp']['patience'],
        save_path=config['mlp']['save_path']
    )
    
    if mlp_model is not None:
        results['mlp'] = {
            'model': mlp_model,
            'train_losses': mlp_train_losses,
            'val_losses': mlp_val_losses,
            'train_accuracies': mlp_train_accs,
            'val_accuracies': mlp_val_accs
        }
        logger.info("✓ MLP model training completed successfully!")
    else:
        logger.warning("✗ MLP model training skipped (CSV file not found or error occurred)")
    
    # 3. Fine-tune BLIP Model
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Fine-tuning BLIP Model for Medical Image Captioning")
    logger.info("="*80 + "\n")
    
    blip_model = fine_tune_blip_model(
        dataset_dir=config['blip']['dataset_dir'],
        epochs=config['blip']['epochs'],
        batch_size=config['blip']['batch_size'],
        learning_rate=config['blip']['learning_rate'],
        max_length=config['blip']['max_length'],
        num_beams=config['blip']['num_beams'],
        temperature=config['blip']['temperature'],
        save_path=config['blip']['save_path']
    )
    
    if blip_model is not None:
        results['blip'] = {'model': blip_model}
        logger.info("✓ BLIP model fine-tuning completed successfully!")
    else:
        logger.warning("✗ BLIP model fine-tuning skipped")
    
    # 4. Generate Evaluation Metrics and Visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Generating Evaluation Metrics and Visualizations")
    logger.info("="*80 + "\n")
    
    if 'cnn' in results and results['cnn']['evaluation']:
        eval_data = results['cnn']['evaluation']
        classes = eval_data['classes']
        all_labels = eval_data['labels']
        all_predictions = eval_data['predictions']
        all_probabilities = np.array(eval_data['probabilities'])
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - CNN Model', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Confusion matrix saved to confusion_matrix.png")
        
        # ROC Curves
        y_true_bin = label_binarize(all_labels, classes=range(len(classes)))
        plt.figure(figsize=(12, 8))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - CNN Model', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ ROC curves saved to roc_curves.png")
        
        # Training Curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CNN Loss
        axes[0, 0].plot(cnn_train_losses, label='Training Loss', linewidth=2)
        axes[0, 0].plot(cnn_val_losses, label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('CNN Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # CNN Accuracy
        axes[0, 1].plot(cnn_train_accs, label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(cnn_val_accs, label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('CNN Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # MLP Loss (if available)
        if 'mlp' in results and mlp_train_losses:
            axes[1, 0].plot(mlp_train_losses, label='Training Loss', linewidth=2)
            axes[1, 0].plot(mlp_val_losses, label='Validation Loss', linewidth=2)
            axes[1, 0].set_title('MLP Training and Validation Loss', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'MLP data not available', ha='center', va='center',
                          transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('MLP Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # MLP Accuracy (if available)
        if 'mlp' in results and mlp_train_accs:
            axes[1, 1].plot(mlp_train_accs, label='Training Accuracy', linewidth=2)
            axes[1, 1].plot(mlp_val_accs, label='Validation Accuracy', linewidth=2)
            axes[1, 1].set_title('MLP Training and Validation Accuracy', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'MLP data not available', ha='center', va='center',
                          transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('MLP Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Training curves saved to training_curves.png")
        
        # Classification Report
        report = classification_report(all_labels, all_predictions, target_names=classes)
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}\n")
        
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        logger.info("✓ Classification report saved to classification_report.txt")
    
    # 5. XAI Demonstration (on a sample image)
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Generating XAI Explanations (Sample)")
    logger.info("="*80 + "\n")
    
    if 'cnn' in results and cnn_model is not None:
        # Find a sample image
        sample_image_path = None
        for class_name in config['cnn']['classes']:
            class_dir = os.path.join(config['cnn']['dataset_dir'], class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                        if 'mask' not in img_name.lower():
                            sample_image_path = os.path.join(class_dir, img_name)
                            break
                if sample_image_path:
                    break
        
        if sample_image_path:
            logger.info(f"Generating XAI explanations for sample image: {sample_image_path}")
            try:
                sample_image = Image.open(sample_image_path).convert('RGB')
                
                # Get prediction
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(sample_image).unsqueeze(0).to(device)
                
                cnn_model.eval()
                with torch.no_grad():
                    output = cnn_model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                
                logger.info(f"Predicted class: {config['cnn']['classes'][predicted_class]}")
                
                # Apply XAI methods
                logger.info("Applying LIME...")
                lime_exp = apply_lime_explanation(cnn_model, sample_image, config['cnn']['classes'], num_samples=500)
                
                logger.info("Applying Grad-CAM...")
                gradcam_map = apply_gradcam(cnn_model, sample_image, predicted_class)
                
                logger.info("Applying Integrated Gradients...")
                ig_map = apply_integrated_gradients(cnn_model, sample_image, predicted_class, steps=30)
                
                # Visualize XAI results
                visualize_xai_results(sample_image, lime_exp, gradcam_map, ig_map, 
                                     config['cnn']['classes'], save_path='xai_results.png')
                logger.info("✓ XAI visualizations saved to xai_results.png")
                
            except Exception as e:
                logger.error(f"Error generating XAI explanations: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("No sample image found for XAI demonstration")
    
    # 6. Save training summary
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Saving Training Summary")
    logger.info("="*80 + "\n")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'models_trained': list(results.keys()),
        'cnn_config': config['cnn'],
        'mlp_config': config['mlp'],
        'blip_config': config['blip']
    }
    
    if 'cnn' in results and results['cnn']['evaluation']:
        summary['cnn_results'] = {
            'test_accuracy': results['cnn']['evaluation']['test_accuracy'],
            'final_train_accuracy': cnn_train_accs[-1] if cnn_train_accs else None,
            'final_val_accuracy': cnn_val_accs[-1] if cnn_val_accs else None
        }
    
    if 'mlp' in results and mlp_train_accs:
        summary['mlp_results'] = {
            'final_train_accuracy': mlp_train_accs[-1] if mlp_train_accs else None,
            'final_val_accuracy': mlp_val_accs[-1] if mlp_val_accs else None
        }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("✓ Training summary saved to training_summary.json")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Models trained: {', '.join(results.keys())}")
    if 'cnn' in results and results['cnn']['evaluation']:
        logger.info(f"CNN Test Accuracy: {results['cnn']['evaluation']['test_accuracy']:.2f}%")
    logger.info("\nGenerated files:")
    logger.info("  - mobilenet_vitamin.pth (CNN model)")
    if 'mlp' in results:
        logger.info("  - cholesterol_mlp.pth (MLP model)")
    if 'blip' in results:
        logger.info("  - blip_medical_captioning/ (BLIP model)")
    logger.info("  - confusion_matrix.png")
    logger.info("  - roc_curves.png")
    logger.info("  - training_curves.png")
    logger.info("  - classification_report.txt")
    logger.info("  - training_summary.json")
    logger.info("  - xai_results.png (if generated)")
    logger.info("  - training.log")
    logger.info("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    # Google Colab specific setup
    try:
        import google.colab
        IN_COLAB = True
        logger.info("Running in Google Colab environment")
        
        # Mount Google Drive if needed
        # from google.colab import drive
        # drive.mount('/content/drive')
        
    except ImportError:
        IN_COLAB = False
        logger.info("Running in local environment")
    
    # Run training
    results = main()