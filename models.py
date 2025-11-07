import os
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize, StandardScaler
import cv2
import shap
from lime.lime_image import LimeImageExplainer
import pandas as pd
import platform
import logging
import streamlit as st
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
if platform.system() == "Darwin" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def clear_mps_cache():
    """Clear MPS cache if on macOS and MPS is available."""
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
        except RuntimeError as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

# Image preprocessing
def preprocess_image(img_path, output_path):
    """Preprocess image for better model performance."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Save processed image
        cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {e}")
        return False

def augment_with_blur(img_path, output_path, blur_radius=2):
    """Create blurred version of image for data augmentation."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (blur_radius*2+1, blur_radius*2+1), blur_radius)
        cv2.imwrite(output_path, blurred)
        return True
    except Exception as e:
        logger.error(f"Error creating blurred image {img_path}: {e}")
        return False

# Dataset class
class VitaminDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
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

# Model loading
@st.cache_resource
def load_cnn_model(classes=None):
    """Load pre-trained MobileNetV2 model."""
    try:
        if classes is None:
            # Default classes if none provided
            classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        
        num_classes = len(classes)
        
        # First try to load the trained model from saved file
        model_path = "mobilenet_vitamin.pth"
        if os.path.exists(model_path):
            try:
                # Load the trained model
                model = models.mobilenet_v2(weights=None)  # Don't load ImageNet weights
                # Modify the classifier for our number of classes
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                
                # Load the trained weights
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                
                # Debug: Check if weights are actually loaded
                first_weight = next(model.parameters())
                weight_mean = first_weight.mean().item()
                weight_std = first_weight.std().item()
                
                logger.info(f"✅ Successfully loaded trained model from {model_path}")
                logger.info(f"🔍 Model weight stats - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
                
                return model
            except Exception as e:
                logger.warning(f"Failed to load trained model from {model_path}: {e}")
        
        # Fallback to untrained model if trained model not available
        logger.warning("No trained model found, loading untrained base model")
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Modify the classifier for our number of classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Initialize the new classifier layer with proper random weights
        torch.nn.init.xavier_uniform_(model.classifier[1].weight)
        torch.nn.init.zeros_(model.classifier[1].bias)
        
        # Debug: Check untrained model weights
        first_weight = next(model.parameters())
        weight_mean = first_weight.mean().item()
        weight_std = first_weight.std().item()
        logger.info(f"🔍 Untrained model weight stats - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
        
        return model.to(device)
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")
        return None

def force_retrain_model(epochs=20, patience=7, verbose=True, classes=None):
    """Force retrain the model and clear cache."""
    try:
        if verbose:
            st.info("🔄 Starting forced model retraining...")
        
        # Clear the cache to force reload
        load_cnn_model.clear()
        
        # Remove existing model file if it exists
        model_path = "mobilenet_vitamin.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
            if verbose:
                st.info(f"🗑️ Removed existing model file: {model_path}")
        
        # Train the model
        if verbose:
            st.info("🚀 Training new model...")
        
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(epochs, patience, verbose=verbose, classes=classes)
        
        if model is not None:
            if verbose:
                st.success("✅ Model retraining completed successfully")
            return True
        else:
            if verbose:
                st.error("❌ Model retraining failed")
            return False
            
    except Exception as e:
        if verbose:
            st.error(f"❌ Error in force retrain: {e}")
        return False

# CSV preprocessing
def load_and_preprocess_csv(csv_path):
    """Load and preprocess cholesterol CSV data."""
    try:
        df = pd.read_csv(csv_path)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Check if 'target' column exists, otherwise use 'num' as target
        if 'target' in df.columns:
            target_column = 'target'
        elif 'num' in df.columns:
            target_column = 'num'
        else:
            logger.error("No target column found in CSV. Expected 'target' or 'num' column.")
            return None, None
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != target_column]
        features = df[feature_columns]
        labels = df[target_column]
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return pd.DataFrame(features_scaled, columns=feature_columns), labels
    except Exception as e:
        logger.error(f"Error preprocessing CSV: {e}")
        return None, None

# MLP model for cholesterol
class CholesterolMLP(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super(CholesterolMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training functions
def train_mlp_model(epochs=20, patience=7):
    """Train MLP model for cholesterol prediction."""
    try:
        # Load data
        csv_path = "dataset/dataset_2190_cholesterol.csv"
        features, labels = load_and_preprocess_csv(csv_path)
        
        if features is None or labels is None:
            st.error("Failed to load CSV data")
            return None, [], [], []
        
        # Determine number of classes
        num_classes = len(labels.unique())
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model with correct number of classes
        model = CholesterolMLP(features.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return model, train_losses, val_losses, train_accuracies, val_accuracies
        
    except Exception as e:
        logger.error(f"Error training MLP model: {e}")
        return None, [], [], [], []

@st.cache_resource
def train_model(epochs=20, patience=7, accum_steps=4, verbose=True, classes=None):
    """Train CNN model for vitamin deficiency detection."""
    try:
        if classes is None:
            # Default classes if none provided
            classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        
        if verbose:
            st.info("🚀 Starting model training...")
            st.info(f"📊 Training Configuration:")
            st.info(f"   - Epochs: {epochs}")
            st.info(f"   - Patience: {patience}")
            st.info(f"   - Accumulation steps: {accum_steps}")
            st.info(f"   - Device: {device}")
            st.info(f"   - Classes: {classes}")
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if verbose:
            st.info("📁 Loading dataset...")
        
        # Create train/val/test splits dynamically
        dataset_dir = "dataset"
        all_images = []
        all_labels = []
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Collect all images and labels
        for class_name in classes:
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(os.path.join(class_path, img_name))
                        all_labels.append(class_to_idx[class_name])
        
        if not all_images:
            st.error("No images found in dataset")
            return None, [], [], [], []
        
        if verbose:
            st.info(f"📊 Dataset loaded: {len(all_images)} total images")
            for i, class_name in enumerate(classes):
                class_count = all_labels.count(i)
                st.info(f"   - {class_name}: {class_count} images")
        
        # Split data into train/val/test
        from sklearn.model_selection import train_test_split
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        if verbose:
            st.info(f"📊 Data split:")
            st.info(f"   - Training: {len(train_images)} images")
            st.info(f"   - Validation: {len(val_images)} images")
            st.info(f"   - Testing: {len(test_images)} images")
        
        # Create custom datasets
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, image_paths, labels, transform=None):
                self.image_paths = image_paths
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
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
        
        train_dataset = CustomDataset(train_images, train_labels, transform=transform)
        val_dataset = CustomDataset(val_images, val_labels, transform=transform)
        test_dataset = CustomDataset(test_images, test_labels, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        if verbose:
            st.info(f"📊 Data loaders created:")
            st.info(f"   - Training batches: {len(train_loader)}")
            st.info(f"   - Validation batches: {len(val_loader)}")
            st.info(f"   - Testing batches: {len(test_loader)}")
        
        # Initialize model
        if verbose:
            st.info("🧠 Loading model...")
        
        model = load_cnn_model(classes=classes)
        if model is None:
            st.error("Failed to load model")
            return None, [], [], [], []
        
        if verbose:
            st.info(f"✅ Model loaded successfully")
            st.info(f"   - Model type: {type(model)}")
            st.info(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            st.info(f"   - Device: {next(model.parameters()).device}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        if verbose:
            st.info("⚙️ Training configuration:")
            st.info(f"   - Loss function: CrossEntropyLoss")
            st.info(f"   - Optimizer: Adam (lr=0.001)")
            st.info(f"   - Scheduler: ReduceLROnPlateau")
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            st.info("🎯 Starting training loop...")
        
        for epoch in range(epochs):
            if verbose:
                st.info(f"🔄 Epoch {epoch+1}/{epochs}")
            
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            # Progress bar for training
            if verbose:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
                loss.backward()
                
                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accum_steps
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress
                if verbose:
                    progress = (i + 1) / len(train_loader)
                    progress_bar.progress(progress)
                    status_text.text(f"Training batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            if verbose:
                st.info(f"✅ Training completed - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            
            # Validation
            if verbose:
                st.info("🔍 Running validation...")
                val_progress_bar = st.progress(0)
                val_status_text = st.empty()
            
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update validation progress
                    if verbose:
                        progress = (i + 1) / len(val_loader)
                        val_progress_bar.progress(progress)
                        val_status_text.text(f"Validation batch {i+1}/{len(val_loader)}")
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            if verbose:
                st.info(f"✅ Validation completed - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
                st.info(f"📈 Epoch {epoch+1} Summary:")
                st.info(f"   - Training Loss: {train_loss:.4f} → Training Accuracy: {train_accuracy:.2f}%")
                st.info(f"   - Validation Loss: {val_loss:.4f} → Validation Accuracy: {val_accuracy:.2f}%")
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                try:
                    torch.save(model.state_dict(), "mobilenet_vitamin.pth")
                    if verbose:
                        st.success(f"💾 Best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
            else:
                patience_counter += 1
                if verbose:
                    st.warning(f"⚠️ No improvement for {patience_counter} epochs (patience: {patience})")
                if patience_counter >= patience:
                    if verbose:
                        st.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Clear progress bars
            if verbose:
                progress_bar.empty()
                status_text.empty()
                val_progress_bar.empty()
                val_status_text.empty()
        
        # Save final model if not already saved
        try:
            torch.save(model.state_dict(), "mobilenet_vitamin.pth")
            if verbose:
                st.success("💾 Final model saved successfully")
        except Exception as e:
            st.error(f"Failed to save final model: {e}")
        
        if verbose:
            st.success("🎉 Training completed successfully!")
            st.info("📊 Final Training Summary:")
            st.info(f"   - Total epochs: {len(train_losses)}")
            st.info(f"   - Best validation loss: {best_val_loss:.4f}")
            st.info(f"   - Final training accuracy: {train_accuracies[-1]:.2f}%")
            st.info(f"   - Final validation accuracy: {val_accuracies[-1]:.2f}%")
        
        return model, train_losses, val_losses, train_accuracies, val_accuracies
        
    except Exception as e:
        st.error(f"❌ Error training CNN model: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None, [], [], [], []

# Evaluation functions
def evaluate_combined_model(cnn_model, mlp_model, test_loader, test_loader_mlp, classes, generate_metrics=True):
    """Evaluate combined CNN and MLP models."""
    try:
        # Evaluate CNN
        cnn_model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn_model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate CNN accuracy
        train_accuracy = 100 * sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(all_labels)
        
        # Store predictions and labels for dashboard
        all_predictions_cnn = all_predictions.copy()
        all_labels_cnn = all_labels.copy()
        
        # Evaluate MLP only if available
        test_accuracy = 0
        if mlp_model is not None and test_loader_mlp is not None:
            mlp_model.eval()
            mlp_predictions = []
            mlp_labels = []
            
            with torch.no_grad():
                for features, labels in test_loader_mlp:
                    features, labels = features.to(device), labels.to(device)
                    outputs = mlp_model(features)
                    _, predicted = torch.max(outputs, 1)
                    
                    mlp_predictions.extend(predicted.cpu().numpy())
                    mlp_labels.extend(labels.cpu().numpy())
            
            # Calculate MLP accuracy
            test_accuracy = 100 * sum(1 for p, l in zip(mlp_predictions, mlp_labels) if p == l) / len(mlp_labels)
        else:
            # If MLP not available, use CNN accuracy for both
            test_accuracy = train_accuracy
        
        cm_path = None
        roc_path = None
        class_report = None
        precisions = []
        recalls = []
        f1_scores = []
        y_true = []
        y_score = []
        
        if generate_metrics:
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC Curves
            y_true = label_binarize(all_labels, classes=range(len(classes)))
            y_score = np.array(all_probabilities)
            
            plt.figure(figsize=(12, 8))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend(loc="lower right")
            roc_path = 'roc_curves.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification Report
            class_report = classification_report(all_labels, all_predictions, target_names=classes)
            
            # Calculate per-class metrics
            for i in range(len(classes)):
                precisions.append(precision_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0))
                recalls.append(recall_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0))
                f1_scores.append(f1_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0))
        
        return train_accuracy, test_accuracy, cm_path, roc_path, class_report, precisions, recalls, f1_scores, y_true, y_score, all_labels_cnn, all_predictions_cnn
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        return 0, 0, None, None, None, [], [], [], [], []

# Explainability functions
def apply_lime(image, model, classes):
    """Apply LIME explainability to image."""
    try:
        explainer = LimeImageExplainer()
        
        def predict_fn(images):
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            batch = torch.stack([transform(img) for img in images]).to(device)
            with torch.no_grad():
                outputs = model(batch)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
        explanation = explainer.explain_instance(
            np.array(image), 
            predict_fn, 
            top_labels=len(classes), 
            hide_color=0, 
            num_samples=1000
        )
        
        # Create enhanced visualization with scales and quantitative metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # LIME explanation with quantitative analysis
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=10, 
            hide_rest=True
        )
        
        # Calculate quantitative metrics
        mask_density = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
        mask_intensity = np.mean(mask[mask > 0]) if np.any(mask > 0) else 0
        mask_max = np.max(mask)
        
        im1 = axes[0, 1].imshow(mask, cmap='hot')
        axes[0, 1].set_title(f'LIME Explanation\n(Density: {mask_density:.1f}%, Max: {mask_max:.3f})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar1.set_label('Importance Score', fontsize=10)
        
        # Overlay on original image
        axes[1, 0].imshow(image)
        overlay = axes[1, 0].imshow(mask, cmap='hot', alpha=0.6)
        axes[1, 0].set_title(f'LIME Overlay\n(Red = High Importance)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        cbar2 = plt.colorbar(overlay, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar2.set_label('Importance Score', fontsize=10)
        
        # Quantitative metrics bar chart
        metrics = ['Density (%)', 'Mean Intensity', 'Max Intensity']
        values = [mask_density, mask_intensity, mask_max * 100]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Quantitative Values', fontweight='bold')
        axes[1, 1].set_title('LIME Analysis Metrics', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(0, max(values) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
        
    except Exception as e:
        logger.error(f"Error applying LIME: {e}")
        return None

def apply_integrated_gradients(image, model, target_class):
    """Apply Integrated Gradients explainability."""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        target = output[0, target_class]
        
        # Backward pass
        target.backward()
        
        # Get gradients
        gradients = input_tensor.grad
        
        # Integrated gradients
        steps = 50
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated_input = alpha * input_tensor
            interpolated_input.requires_grad_(True)
            
            output = model(interpolated_input)
            target = output[0, target_class]
            target.backward()
            
            integrated_gradients += interpolated_input.grad / steps
        
        # Create visualization
        ig_map = integrated_gradients.squeeze().abs().mean(dim=0).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.imshow(ig_map, alpha=0.6, cmap='hot')
        plt.colorbar()
        plt.title('Integrated Gradients')
        plt.axis('off')
        
        # Save to bytes
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
        
    except Exception as e:
        logger.error(f"Error applying Integrated Gradients: {e}")
        return None

def apply_gradcam(image, model, target_class):
    """Apply Grad-CAM explainability."""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Register hooks
        activations = None
        gradients = None
        
        def activation_hook(module, input, output):
            nonlocal activations
            activations = output
        
        def gradient_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        # Register hooks on the last convolutional layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            logger.error("No convolutional layer found for Grad-CAM")
            return None
        
        target_layer.register_forward_hook(activation_hook)
        target_layer.register_backward_hook(gradient_hook)
        
        # Forward pass
        output = model(input_tensor)
        target = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target.backward()
        
        # Calculate Grad-CAM
        if activations is not None and gradients is not None:
            weights = torch.mean(gradients, dim=[2, 3])
            cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            
            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Create enhanced visualization with scales and quantitative metrics
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Grad-CAM heatmap with quantitative analysis
            cam_max = np.max(cam)
            cam_mean = np.mean(cam)
            cam_std = np.std(cam)
            high_attention_pixels = np.sum(cam > cam_mean + cam_std)
            total_pixels = cam.shape[0] * cam.shape[1]
            high_attention_percentage = (high_attention_pixels / total_pixels) * 100
            
            im1 = axes[0, 1].imshow(cam, cmap='jet')
            axes[0, 1].set_title(f'Grad-CAM Heatmap\n(Max: {cam_max:.3f}, Mean: {cam_mean:.3f})', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar1.set_label('Attention Score (0-1)', fontsize=10)
            
            # Overlay on original image
            axes[1, 0].imshow(image)
            overlay = axes[1, 0].imshow(cam, alpha=0.6, cmap='jet')
            axes[1, 0].set_title(f'Grad-CAM Overlay\n(Red = High Attention)', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            cbar2 = plt.colorbar(overlay, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar2.set_label('Attention Score (0-1)', fontsize=10)
            
            # Quantitative metrics bar chart
            metrics = ['Max Attention', 'Mean Attention', 'High Attention (%)', 'Std Dev']
            values = [cam_max, cam_mean, high_attention_percentage, cam_std]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.8)
            axes[1, 1].set_ylabel('Quantitative Values', fontweight='bold')
            axes[1, 1].set_title('Grad-CAM Analysis Metrics', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylim(0, max(values) * 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to bytes
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
        
        return None
        
    except Exception as e:
        logger.error(f"Error applying Grad-CAM: {e}")
        return None

# Plotting functions
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, mlp_train_losses, mlp_val_losses, mlp_train_accuracies, mlp_val_accuracies, classes, y_true, y_score, all_labels=None, all_predictions=None):
    """Create and save comprehensive metric plots."""
    plot_paths = []
    
    try:
        # 1. Training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CNN Loss
        if train_losses and val_losses:
            axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
            axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('CNN Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].legend(fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
        
        # CNN Accuracy
        if train_accuracies and val_accuracies:
            axes[0, 1].plot(train_accuracies, label='Training Accuracy', linewidth=2)
            axes[0, 1].plot(val_accuracies, label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title('CNN Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
        
        # MLP Loss - only plot if data is available
        if mlp_train_losses and mlp_val_losses:
            axes[1, 0].plot(mlp_train_losses, label='Training Loss', linewidth=2)
            axes[1, 0].plot(mlp_val_losses, label='Validation Loss', linewidth=2)
            axes[1, 0].set_title('MLP Training and Validation Loss', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'MLP data not available', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('MLP Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # MLP Accuracy - only plot if data is available
        if mlp_train_accuracies and mlp_val_accuracies:
            axes[1, 1].plot(mlp_train_accuracies, label='Training Accuracy', linewidth=2)
            axes[1, 1].plot(val_accuracies, label='Validation Accuracy', linewidth=2)
            axes[1, 1].set_title('MLP Training and Validation Accuracy', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'MLP data not available', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('MLP Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plot_paths.append('training_curves.png')
        plt.close()
        
        # 2. ROC Curves
        if len(y_true) > 0 and len(y_score) > 0:
            plt.figure(figsize=(12, 8))
            roc_aucs = []
            
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                roc_aucs.append(roc_auc)
                plt.plot(fpr, tpr, linewidth=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
            plot_paths.append('roc_curves.png')
            plt.close()
        
        # 3. Precision-Recall curves
        if len(y_true) > 0 and len(y_score) > 0:
            plt.figure(figsize=(12, 8))
            for i in range(len(classes)):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
                plt.plot(recall, precision, linewidth=2, label=f'{classes[i]}')
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
            plot_paths.append('precision_recall_curves.png')
            plt.close()
        
        # 4. Confusion Matrix (if we have predictions)
        if all_labels is not None and all_predictions is not None:
            try:
                cm = confusion_matrix(all_labels, all_predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
                plt.ylabel('True Label', fontsize=12)
                plt.xlabel('Predicted Label', fontsize=12)
                plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
                plot_paths.append('confusion_matrix.png')
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create confusion matrix: {e}")
        
        # 5. Performance Summary Dashboard
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different metrics
        gs = plt.GridSpec(2, 3, figure=plt.gcf())
        
        # Training progress
        ax1 = plt.subplot(gs[0, 0])
        if train_accuracies and val_accuracies:
            ax1.plot(train_accuracies, label='Train', linewidth=2)
            ax1.plot(val_accuracies, label='Validation', linewidth=2)
            ax1.set_title('Training Progress', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Loss progress
        ax2 = plt.subplot(gs[0, 1])
        if train_losses and val_losses:
            ax2.plot(train_losses, label='Train', linewidth=2)
            ax2.plot(val_losses, label='Validation', linewidth=2)
            ax2.set_title('Loss Progress', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Final accuracy comparison
        ax3 = plt.subplot(gs[0, 2])
        if train_accuracies and val_accuracies:
            final_train_acc = train_accuracies[-1] if train_accuracies else 0
            final_val_acc = val_accuracies[-1] if val_accuracies else 0
            bars = ax3.bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                          color=['#4299e1', '#48bb78'], alpha=0.7)
            ax3.set_title('Final Accuracy', fontweight='bold')
            ax3.set_ylabel('Accuracy (%)')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # ROC AUC summary
        ax4 = plt.subplot(gs[1, :2])
        if len(y_true) > 0 and len(y_score) > 0:
            roc_aucs = []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                roc_aucs.append(roc_auc)
            
            bars = ax4.bar(classes, roc_aucs, color='#ed8936', alpha=0.7)
            ax4.set_title('ROC AUC by Class', fontweight='bold')
            ax4.set_ylabel('AUC Score')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Model info
        ax5 = plt.subplot(gs[1, 2])
        ax5.axis('off')
        info_text = f"""
        Model Performance Summary
        
        Training Epochs: {len(train_accuracies) if train_accuracies else 0}
        Final Training Accuracy: {train_accuracies[-1] if train_accuracies else 0:.2f}%
        Final Validation Accuracy: {val_accuracies[-1] if val_accuracies else 0:.2f}%
        Number of Classes: {len(classes)}
        
        Classes: {', '.join(classes)}
        """
        ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
        plot_paths.append('performance_summary.png')
        plt.close()
        
        logger.info(f"✅ Generated {len(plot_paths)} visualization files: {plot_paths}")
        return plot_paths
        
    except Exception as e:
        logger.error(f"Error creating metric plots: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def create_evaluation_dashboard(y_true, y_score, all_labels, all_predictions, classes, train_accuracies, val_accuracies, train_losses, val_losses):
    """Create comprehensive evaluation dashboard with separate visualizations."""
    dashboard_paths = {}
    
    try:
        # Check if we have enough data to create visualizations
        if not y_true or not y_score or not all_labels or not all_predictions:
            logger.warning("Insufficient data for evaluation dashboard")
            return dashboard_paths
        # 1. ROC Curves
        if len(y_true) > 0 and len(y_score) > 0:
            plt.figure(figsize=(12, 8))
            roc_aucs = []
            
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                roc_aucs.append(roc_auc)
                plt.plot(fpr, tpr, linewidth=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title('ROC Curves - Model Performance', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add average AUC
            avg_auc = np.mean(roc_aucs)
            plt.text(0.02, 0.98, f'Average AUC: {avg_auc:.3f}', transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('roc_curves_dashboard.png', dpi=300, bbox_inches='tight')
            dashboard_paths['roc_curves'] = 'roc_curves_dashboard.png'
            plt.close()
        
        # 2. Precision Table and Metrics
        if len(y_true) > 0 and len(y_score) > 0:
            # Calculate comprehensive metrics
            metrics_data = []
            for i in range(len(classes)):
                precision = precision_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0)
                recall = recall_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0)
                f1 = f1_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0)
                
                # Calculate AUC for this class
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                auc_score = auc(fpr, tpr)
                
                metrics_data.append({
                    'Class': classes[i],
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'AUC': auc_score
                })
            
            # Create precision table visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Table
            table_data = [[d['Class'], f"{d['Precision']:.3f}", f"{d['Recall']:.3f}", 
                          f"{d['F1-Score']:.3f}", f"{d['AUC']:.3f}"] for d in metrics_data]
            
            table = ax1.table(cellText=table_data,
                            colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(5):
                    if i == 0:  # Header row
                        table[(i, j)].set_facecolor('#4CAF50')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    else:  # Data rows
                        if j == 0:  # Class names
                            table[(i, j)].set_facecolor('#E8F5E8')
                        else:  # Metric values
                            value = float(table_data[i-1][j])
                            if value >= 0.8:
                                table[(i, j)].set_facecolor('#C8E6C9')  # Light green
                            elif value >= 0.6:
                                table[(i, j)].set_facecolor('#FFF9C4')  # Light yellow
                            else:
                                table[(i, j)].set_facecolor('#FFCDD2')  # Light red
            
            ax1.set_title('Precision, Recall, F1-Score & AUC Table', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Bar chart of metrics
            x = np.arange(len(classes))
            width = 0.2
            
            precisions = [d['Precision'] for d in metrics_data]
            recalls = [d['Recall'] for d in metrics_data]
            f1_scores = [d['F1-Score'] for d in metrics_data]
            aucs = [d['AUC'] for d in metrics_data]
            
            ax2.bar(x - 1.5*width, precisions, width, label='Precision', color='#4CAF50', alpha=0.8)
            ax2.bar(x - 0.5*width, recalls, width, label='Recall', color='#2196F3', alpha=0.8)
            ax2.bar(x + 0.5*width, f1_scores, width, label='F1-Score', color='#FF9800', alpha=0.8)
            ax2.bar(x + 1.5*width, aucs, width, label='AUC', color='#9C27B0', alpha=0.8)
            
            ax2.set_xlabel('Classes', fontweight='bold')
            ax2.set_ylabel('Score', fontweight='bold')
            ax2.set_title('Performance Metrics by Class', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig('precision_table_dashboard.png', dpi=300, bbox_inches='tight')
            dashboard_paths['precision_table'] = 'precision_table_dashboard.png'
            plt.close()
        
        # 3. Accuracy Graph
        if train_accuracies and val_accuracies:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Training vs Validation Accuracy
            epochs = range(1, len(train_accuracies) + 1)
            ax1.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Training Accuracy', marker='o')
            ax1.plot(epochs, val_accuracies, 'r-', linewidth=2, label='Validation Accuracy', marker='s')
            ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontweight='bold')
            ax1.set_ylabel('Accuracy (%)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Training vs Validation Loss
            if train_losses and val_losses:
                ax2.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o')
                ax2.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s')
                ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch', fontweight='bold')
                ax2.set_ylabel('Loss', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('accuracy_graph_dashboard.png', dpi=300, bbox_inches='tight')
            dashboard_paths['accuracy_graph'] = 'accuracy_graph_dashboard.png'
            plt.close()
        
        # 4. Confusion Matrix
        if len(all_labels) > 0 and len(all_predictions) > 0:
            cm = confusion_matrix(all_labels, all_predictions)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Confusion Matrix Heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1)
            ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax1.set_ylabel('True Label', fontweight='bold')
            ax1.set_xlabel('Predicted Label', fontweight='bold')
            
            # Normalized Confusion Matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2)
            ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
            ax2.set_ylabel('True Label', fontweight='bold')
            ax2.set_xlabel('Predicted Label', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('confusion_matrix_dashboard.png', dpi=300, bbox_inches='tight')
            dashboard_paths['confusion_matrix'] = 'confusion_matrix_dashboard.png'
            plt.close()
        
        # 5. Additional Performance Summary
        if len(y_true) > 0 and len(y_score) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate overall metrics
            overall_accuracy = sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(all_labels)
            macro_precision = np.mean([precision_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0) for i in range(len(classes))])
            macro_recall = np.mean([recall_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0) for i in range(len(classes))])
            macro_f1 = np.mean([f1_score(y_true[:, i], y_score[:, i] > 0.5, zero_division=0) for i in range(len(classes))])
            
            metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
            values = [overall_accuracy, macro_precision, macro_recall, macro_f1]
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8)
            ax.set_title('Overall Model Performance Summary', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('performance_summary_dashboard.png', dpi=300, bbox_inches='tight')
            dashboard_paths['performance_summary'] = 'performance_summary_dashboard.png'
            plt.close()
        
        return dashboard_paths
        
    except Exception as e:
        logger.error(f"Error creating evaluation dashboard: {e}")
        return {} 