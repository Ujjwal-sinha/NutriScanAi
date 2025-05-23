# Line 1: Import required libraries
import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
from datetime import datetime
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF
from torchvision.models import MobileNet_V2_Weights
from langchain_core.caches import BaseCache
import tempfile
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
import shap
from lime.lime_image import LimeImageExplainer
import shutil

# Line 29: Set Streamlit page configuration
st.set_page_config(
    page_title="NutriScan AI - Vitamin Deficiency Detector",
    layout="centered",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Line 35: Define function to load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using fallback inline styles.")
        st.markdown("""
        <style>
        .gradient-text {
            background: -webkit-linear-gradient(left, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .download-btn {
            display: inline-block;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            text-decoration: none;
            color: white;
            background: linear-gradient(to right, #4CAF50, #2196F3);
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
            width: 100%;
            box-sizing: border-box;
        }
        .download-btn:hover {
            background: linear-gradient(to right, #45a049, #1e88e5);
            transform: scale(1.05);
        }
        .download-btn:active {
            transform: scale(0.95);
        }
        </style>
        """, unsafe_allow_html=True)

# Line 84: Load custom CSS
local_css("style.css")

# Line 86: Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Line 92: Set device for PyTorch (M1 Mac with MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
st.write(f"Using device: {device}")

# Line 95: Cache BLIP models
@st.cache_resource
def load_models():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load BLIP models: {e}")
        return None, None

# Line 104: Load BLIP models
processor, model = load_models()
if not processor or not model:
    st.error("Critical error: BLIP models failed to load. Please try again later.")
    st.stop()

# Line 109: Define optimized image processing function
def describe_image(image: Image.Image) -> str:
    try:
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        inputs = processor(images=image, return_tensors="pt")
        # Convert to float32 explicitly
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32).to(device)
        out = model.generate(pixel_values=inputs["pixel_values"], max_length=100, num_beams=5)
        caption = processor.decode(out[0], skip_special_tokens=True)
        torch.mps.empty_cache()
        return caption
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return ""




# Line 121: Define efficient prompt template
EFFICIENT_PROMPT_TEMPLATE = """
As a board-certified nutritionist with 20+ years experience, analyze this medical image showing: {caption}
Additional patient context: {context}

Generate a comprehensive deficiency analysis with:

1. **Primary Deficiency Identification** (Most likely 1-2 deficiencies)
   - Vitamin: 
   - Confidence Level: (High/Medium/Low)
   - Key Visual Indicators: 

2. **Detailed Analysis**
   - Physiological Mechanism: (How deficiency causes these symptoms)
   - Differential Diagnosis: (Other possible explanations)

3. **Evidence-Based Recommendations**
   - Top 5 Food Sources (bioavailable forms)
   - Supplement Protocol (if warranted):
     - Type:
     - Dosage:
     - Duration:
   - Synergistic Nutrients:

4. **Clinical Considerations**
   - Expected Timeline for Improvement
   - When to Consult Physician
   - Red Flag Symptoms

5. **Patient Education**
   - Simple Dietary Changes
   - Preparation Tips
   - Common Pitfalls

Format using markdown with clear headings. Be concise but thorough.
"""

# Line 161: Define optimized LLM query function
def query_langchain(prompt: str, predicted_class: str, confidence: float) -> str:
    try:
        chat = ChatGroq(
            temperature=0.3,
            model_name="llama3-70b-8192",
            groq_api_key=GROQ_API_KEY,
            request_timeout=120,
            max_tokens=4000
        )
        # Rebuild the model to resolve BaseCache dependency
        ChatGroq.model_rebuild()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI medical nutrition specialist. Provide accurate, evidence-based analysis."),
            ("user", EFFICIENT_PROMPT_TEMPLATE)
        ])
        chain = prompt_template | chat | StrOutputParser()
        result = chain.invoke({"caption": f"{prompt}. Predicted deficiency: {predicted_class} (Confidence: {confidence:.2f})", "context": user_context or "None provided"})
        torch.mps.empty_cache()  # Clear memory after LLM
        return result
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")
        return ""

# Line 178: Define enhanced PDF generator class
class MedicalPDF(FPDF):
    def __init__(self, patient_info=""):
        super().__init__()
        self.patient_info = patient_info
        self.toc = []
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font('Helvetica', '', 12)
        self.set_author("NutriScan AI")
        self.set_title("NutriScan AI Deficiency Report")
        self.set_subject("Vitamin Deficiency Analysis")
    
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(76, 175, 80)
            self.cell(0, 10, 'NutriScan AI Deficiency Report', 0, 1, 'C')
            self.set_line_width(0.5)
            self.set_draw_color(33, 150, 243)
            self.line(10, 20, 200, 20)
            self.ln(10)
    
    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Page {self.page_no() - 1} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
            self.set_font('Helvetica', 'I', 6)
            self.set_text_color(200, 200, 200)
            self.set_xy(10, 270)
            self.cell(0, 10, 'Powered by NutriScan AI', 0, 0, 'L')
    
    def cover_page(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(76, 175, 80)
        self.cell(0, 20, 'NutriScan AI', 0, 1, 'C')
        self.set_font('Helvetica', '', 16)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'Vitamin Deficiency Analysis Report', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)
        if self.patient_info:
            self.multi_cell(0, 8, f'Patient Information:\n{self.patient_info}')
        self.ln(10)
        self.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 8, 'For educational purposes only', 0, 1, 'C')
    
    def table_of_contents(self):
        self.add_page()
        self.toc.append(("Table of Contents", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(76, 175, 80)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'C')
        self.ln(10)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)
        for title, page in self.toc:
            self.cell(0, 8, f'{title} {"." * (50 - len(title))} {page}', ln=1)
        self.ln(10)
    
    def add_image(self, image_path, width=180):
        try:
            self.image(image_path, x=(self.w - width)/2, w=width)
            self.ln(10)
        except Exception as e:
            st.error(f"Failed to add image to PDF: {e}")
    
    def add_section(self, title, body):
        self.toc.append((title, self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        lines = body.split('\n')
        in_list = False
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                if not in_list:
                    in_list = True
                    self.set_left_margin(15)
                self.multi_cell(0, 8, f'• {line[2:]}'.encode('latin1', 'ignore').decode('latin1'))
            elif line.startswith('|'):
                self.set_left_margin(10)
                self.create_table(line)
            else:
                if in_list:
                    in_list = False
                    self.set_left_margin(10)
                self.multi_cell(0, 8, line.encode('latin1', 'ignore').decode('latin1'))
            self.ln(2)
        self.set_left_margin(10)
        self.ln(5)
    
    def create_table(self, line):
        if line.startswith('|'):
            cols = [col.strip() for col in line.split('|')[1:-1]]
            col_width = (self.w - 20) / len(cols)
            self.set_font('Helvetica', 'B', 11)
            for col in cols:
                self.cell(col_width, 10, col.encode('latin1', 'ignore').decode('latin1'), border=1)
            self.ln()
            self.set_font('Helvetica', '', 11)
    
    def add_summary(self, report):
        self.add_page()
        self.toc.append(("Executive Summary", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'Executive Summary', 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        summary = "This report provides an AI-generated analysis of potential vitamin deficiencies based on the provided medical image. Key findings include:\n"
        try:
            if "1. **Primary Deficiency Identification**" in report:
                primary = report.split("2. **Detailed Analysis**")[0].split("1. **Primary Deficiency Identification**")[1]
                summary += f"- {primary.strip()[:200]}...\n"
            else:
                summary += "- Primary deficiency information not found.\n"
            if "3. **Evidence-Based Recommendations**" in report:
                recommendations = report.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0]
                summary += f"- Recommendations: {recommendations.strip()[:200]}...\n"
            else:
                summary += "- Recommendations not found.\n"
            summary += "Refer to the detailed sections for comprehensive insights."
        except IndexError:
            summary += "- Unable to generate summary due to report structure. Please review detailed sections."
        self.multi_cell(0, 8, summary.encode('latin1', 'ignore').decode('latin1'))
        self.ln(10)
    

    
    def add_explainability(self, lime_path, ig_path, gradcam_path):
         self.add_page()
         self.toc.append(("Explainability Analysis", self.page_no()))
         self.set_font('Helvetica', 'B', 14)
         self.set_text_color(33, 150, 243)
         self.cell(0, 10, 'Explainability Analysis', 0, 1)
         self.ln(5)
         self.set_font('Helvetica', '', 11)
         self.set_text_color(0, 0, 0)
         self.cell(0, 8, "LIME (Highlighted Regions):", 0, 1)
         self.image(lime_path, w=180)
         self.ln(5)
         self.cell(0, 8, "Integrated Gradients (Pixel Importance):", 0, 1)
         self.image(ig_path, w=180)
         self.ln(5)
         self.cell(0, 8, "Grad-CAM (Influential Regions):", 0, 1)
         self.image(gradcam_path, w=180)

# Line 343: Define gradient text function
def gradient_text(text, color1, color2):
    return f"""
    <style>
    .gradient-text {{
        background: -webkit-linear-gradient(left, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }}
    </style>
    <div class="gradient-text">{text}</div>
    """

# Line 353: Prepare dataset
dataset_dir = "dataset"  # Update this path, e.g., "/Users/yourname/dataset"
output_dir = "split_dataset"
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
if not classes:
    st.error("No classes found in dataset directory. Please check dataset structure.")
    st.stop()

# Line 360: Expected classes: Vitamin A, B, C, D, E, and optionally No Deficiency
expected_classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]
if "No Deficiency" in classes:
    expected_classes.append("No Deficiency")
if sorted(classes) != sorted(expected_classes):
    st.warning(f"Expected classes {expected_classes}, but found {classes}. Proceeding with available classes.")

# Line 366: Split dataset
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    images = [os.path.join(dataset_dir, cls, f) for f in os.listdir(os.path.join(dataset_dir, cls)) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    for img in train_imgs:
        shutil.copy(img, os.path.join(output_dir, "train", cls, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(output_dir, "val", cls, os.path.basename(img)))
    for img in test_imgs:
        shutil.copy(img, os.path.join(output_dir, "test", cls, os.path.basename(img)))

# Line 381: Preprocess images (reduced size for 8GB memory)
def preprocess_image(img_path, output_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img.save(output_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

for split in ["train", "val", "test"]:
    for cls in classes:
        for img_name in os.listdir(os.path.join(output_dir, split, cls)):
            img_path = os.path.join(output_dir, split, cls, img_name)
            preprocess_image(img_path, img_path)

# Line 394: Define dataset class
class VitaminDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        for cls in self.classes:
            for img_name in os.listdir(os.path.join(root_dir, cls)):
                self.images.append(os.path.join(root_dir, cls, img_name))
                self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Line 418: Data transforms with augmentation (updated for 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to float32 and scales to [0,1]
    transforms.Lambda(lambda x: x.to(torch.float32)),  # Force float32
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.contiguous())  # Ensure proper memory layout
])

# Line 425: Load datasets
train_dataset = VitaminDataset(os.path.join(output_dir, "train"), transform=transform)
val_dataset = VitaminDataset(os.path.join(output_dir, "val"), transform=transform)
test_dataset = VitaminDataset(os.path.join(output_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced for 8GB memory
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# Line 432: Train MobileNetV2
model_cnn = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model_cnn.classifier[1] = nn.Linear(model_cnn.last_channel, len(classes))
model_cnn = model_cnn.to(device).float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

# Line 438: Define training function
@st.cache_resource
def train_model():
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    for epoch in range(5):  # Reduced epochs for faster training
        model_cnn.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        st.write(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        model_cnn.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_cnn(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        st.write(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model_cnn.state_dict(), "mobilenet_vitamin.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            st.write("Early stopping")
            break
        
        torch.mps.empty_cache()  # Clear memory after each epoch
    
    return model_cnn



# Line 478: Updated evaluate_model function
# Line 478: Updated evaluate_model function
def evaluate_model(model, test_loader, classes, generate_metrics=True):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    train_correct, train_total = 0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    train_accuracy = 100 * train_correct / train_total
    test_accuracy = 100 * np.sum(y_true == y_pred) / len(y_true)
    
    cm_path = None
    roc_path = None
    class_report = None
    
    if generate_metrics:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        cm_path = "confusion_matrix.png"  # Assign the path after saving
        
        class_report = classification_report(y_true, y_pred, target_names=classes)
        
        y_true_bin = label_binarize(y_true, classes=range(len(classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure(figsize=(8, 6))
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.close()
        roc_path = "roc_curve.png"  # Assign the path after saving
    
    torch.mps.empty_cache()  # Clear memory after evaluation
    return train_accuracy, test_accuracy, cm_path, roc_path, class_report




# Line 525: Explainability functions
def apply_lime(image, model, classes):
    explainer = LimeImageExplainer()
    def predict_fn(images):
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Ensure float32
        images = images / 255.0  # LIME passes images in [0, 255], normalize to [0, 1]
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    image_np = np.array(image.resize((224, 224)))
    explanation = explainer.explain_instance(image_np, predict_fn, top_labels=2, num_samples=500)  # Reduced for 8GB memory
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True)
    plt.imshow(temp)
    plt.savefig("lime_output.png")
    plt.close()
    torch.mps.empty_cache()  # Clear memory after LIME
    return "lime_output.png"


# Line 538: Updated Integrated Gradients function
def apply_integrated_gradients(image, model, target_class):
    model.eval()
    # Create image tensor as a leaf variable
    image_tensor = transform(image).unsqueeze(0).to(device).float()
    image_tensor = image_tensor.detach()  # Detach from computation graph
    image_tensor.requires_grad = True  # Now a leaf tensor, so this is safe
    
    # Define a baseline (black image)
    baseline = torch.zeros_like(image_tensor).to(device)
    baseline = baseline.detach()
    
    # Number of steps for integration
    steps = 50
    integrated_gradients = torch.zeros_like(image_tensor)
    
    # Compute gradients at each step
    for alpha in torch.linspace(0, 1, steps, device=device):
        interpolated_input = baseline + alpha * (image_tensor - baseline)
        interpolated_input = interpolated_input.detach()
        interpolated_input.requires_grad = True  # Now a leaf tensor
        outputs = model(interpolated_input)
        model.zero_grad()
        # Use torch.autograd.grad to compute gradients directly
        gradients = torch.autograd.grad(outputs[0, target_class], interpolated_input, create_graph=False)[0]
        integrated_gradients += gradients / steps
    
    # Compute the attribution
    attribution = integrated_gradients * (image_tensor - baseline)
    attribution = attribution.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    attribution = np.abs(attribution).sum(axis=-1)  # Sum across channels
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
    attribution = np.uint8(255 * attribution)
    attribution = cv2.applyColorMap(attribution, cv2.COLORMAP_JET)
    
    image_np = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    attribution = attribution.astype(np.float32) / 255.0
    superimposed_img = attribution * 0.4 + image_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1) * 255
    superimposed_img = superimposed_img.astype(np.uint8)
    cv2.imwrite("ig_output.jpg", superimposed_img)
    torch.mps.empty_cache()
    return "ig_output.jpg"

# Line 553: Updated Grad-CAM function
# Line 553: Updated Grad-CAM function using hooks
# Line 553: Updated Grad-CAM function
def apply_gradcam(image, model, target_class):
    model.eval()
    # Apply initial transform up to ToTensor to get a leaf tensor
    initial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = initial_transform(image).unsqueeze(0).to(device).float()
    image_tensor = image_tensor.clone()
    image_tensor.requires_grad = True  # This is now a leaf tensor
    
    # Apply remaining transforms (normalization) without breaking requires_grad
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)
    
    # Register hooks to capture gradients and activations
    activations = None
    gradients = None
    
    def activation_hook(module, input, output):
        nonlocal activations
        activations = output.clone().detach()
    
    def gradient_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].clone().detach()
    
    # Attach hooks to the last convolutional layer
    target_layer = model.features[-1]
    handle1 = target_layer.register_forward_hook(activation_hook)
    handle2 = target_layer.register_backward_hook(gradient_hook)
    
    # Forward and backward pass
    outputs = model(image_tensor)
    model.zero_grad()
    outputs[0, target_class].backward()
    
    # Detach hooks
    handle1.remove()
    handle2.remove()
    
    # Compute Grad-CAM heatmap
    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
    weighted_activations = activations * pooled_gradients
    heatmap = torch.mean(weighted_activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    image_np = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    heatmap = heatmap.astype(np.float32) / 255.0
    superimposed_img = heatmap * 0.4 + image_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1) * 255
    superimposed_img = superimposed_img.astype(np.uint8)
    cv2.imwrite("gradcam_output.jpg", superimposed_img)
    torch.mps.empty_cache()
    return "gradcam_output.jpg"

# Line 576: Main UI setup
st.markdown(gradient_text("NutriScan AI", "#4CAF50", "#2196F3"), unsafe_allow_html=True)
st.markdown("### AI-Powered Vitamin Deficiency Detection from Medical Images")
st.markdown("**Disclaimer**: This tool is for educational purposes only. Consult a healthcare professional for medical advice.")

# Line 580: Updated sidebar content
# Line 580: Updated sidebar content
with st.sidebar:
    st.header("About")
    st.markdown("""
    **NutriScan AI** combines:
    - Computer vision
    - Nutritional biochemistry
    - Clinical medicine
    
    For accurate deficiency detection.
    """)
    st.divider()
    st.subheader("Model Evaluation")
    if st.button("Train and Evaluate Model"):
        with st.spinner("Training model..."):
            model_cnn = train_model()
            train_accuracy, test_accuracy, cm_path, roc_path, class_report = evaluate_model(model_cnn, test_loader, classes, generate_metrics=True)
            if cm_path and os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix")
            else:
                st.warning("Confusion Matrix image not available.")
            if roc_path and os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curves")
            else:
                st.warning("ROC Curves image not available.")
            st.text(f"Training Accuracy: {train_accuracy:.2f}%")
            st.text(f"Testing Accuracy: {test_accuracy:.2f}%")
            if class_report:
                st.text("Classification Report:\n" + class_report)
            else:
                st.warning("Classification Report not available.")
            st.session_state.model_trained = True


# Line 598: Image upload with preview
col1, col2 = st.columns([3, 2])
with col1:
    img_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png"],
        help="Clear photos of skin, nails, tongue, or eyes work best",
        key="image_uploader"
    )

with col2:
    if img_file:
        try:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Preview", use_column_width=True)
        except UnidentifiedImageError:
            st.warning("Invalid image file. Please upload a valid JPG, JPEG, or PNG file.")

# Line 612: Enhanced input form
with st.expander("➕ Additional Clinical Context"):
    user_context = st.text_area(
        "Patient Information",
        placeholder="Age, symptoms duration, medical history, current medications...",
        height=100
    )
    clinical_factors = st.multiselect(
        "Relevant Factors",
        ["Diabetes", "Pregnancy", "Vegan Diet", "Malabsorption", "Alcohol Use"],
        placeholder="Select applicable conditions"
    )

# Line 622: Analysis button with reset
col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze with NutriScan AI", type="primary", use_container_width=True, key="analyze_button"):
        if not img_file:
            st.warning("Please upload an image")
            st.stop()

        if not st.session_state.get("model_trained", False):
            st.warning("Please train the model first using the sidebar button.")
            st.stop()

        with st.spinner("Processing image with AI..."):
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Image processing with BLIP
                progress_bar.progress(20, text="Analyzing visual patterns")
                image = Image.open(img_file).convert("RGB")
                caption = describe_image(image)
                
                if not caption:
                    st.stop()

                # Step 2: CNN classification
                progress_bar.progress(40, text="Classifying deficiency")
                img_tensor = transform(image).unsqueeze(0).to(device).float()  # Ensure float32
                with torch.no_grad():
                    cnn_output = model_cnn(img_tensor)
                    probabilities = torch.softmax(cnn_output, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    predicted_class = classes[predicted_idx]
                    confidence = probabilities[0, predicted_idx].item()
                
                # Step 3: Clinical correlation with LLM
                progress_bar.progress(50, text="Correlating with clinical data")
                result = query_langchain(caption, predicted_class, confidence)
                
                if not result:
                    st.stop()

                # Step 4: Display results
                progress_bar.progress(90, text="Formatting report")
                
                with st.container():
                    st.subheader("🔬 Comprehensive Deficiency Analysis")
                    st.write(f"**CNN Prediction**: {predicted_class} (Confidence: {confidence:.2f})")
                    tab1, tab2, tab3, tab4 = st.tabs(["Primary Findings", "Recommendations", "Clinical Notes", "Explainability"])
                    
                    with tab1:
                        st.markdown(result.split("2. **Detailed Analysis**")[0])
                    
                    with tab2:
                        if "3. **Evidence-Based Recommendations**" in result:
                            st.markdown(result.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0])
                    
                    with tab3:
                        if "4. **Clinical Considerations**" in result:
                            st.markdown(result.split("4. **Clinical Considerations**")[1])
                    
                    with tab4:
                        st.image(apply_lime(image, model_cnn, classes), caption="LIME: Highlighted regions")
                        st.image(apply_integrated_gradients(image, model_cnn, predicted_idx), caption="Integrated Gradients: Pixel importance")
                        st.image(apply_gradcam(image, model_cnn, predicted_idx), caption="Grad-CAM: Influential regions")
                
                st.session_state.report_data = {
                    "image": image,
                    "report": result,
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "timestamp": datetime.now()
                }
                
                progress_bar.progress(100, text="Analysis complete")
                st.success("✓ Report generated")
                
            except Exception as e:
                progress_bar.empty()
                st.error(f"Analysis failed: {str(e)}")

with col2:
    if st.button("Reset", use_container_width=True, key="reset_button"):
        st.session_state.clear()
        st.experimental_rerun()



# Line 685: Updated report generation section
if 'report_data' in st.session_state:
    st.divider()
    st.subheader("Report Options")
    
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True, key="pdf_button"):
        with st.spinner("Generating professional report..."):
            try:
                patient_info = st.session_state.get('report_data', {}).get('patient_info', user_context or "Not provided")
                pdf = MedicalPDF(patient_info=patient_info)
                
                pdf.cover_page()
                pdf.add_summary(st.session_state.report_data["report"])
                pdf.table_of_contents()
                
                pdf.add_page()
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    st.session_state.report_data["image"].save(tmp.name, quality=90)
                    pdf.add_image(tmp.name)
                    os.unlink(tmp.name)
                
                report = st.session_state.report_data["report"]
                sections = [
                    ("Clinical Findings", report.split("2. **Detailed Analysis**")[0]),
                    ("Detailed Analysis", report.split("2. **Detailed Analysis**")[1].split("3. **Evidence-Based Recommendations**")[0] if "2. **Detailed Analysis**" in report else ""),
                    ("Treatment Plan", report.split("3. **Evidence-Based Recommendations**")[1] if "3. **Evidence-Based Recommendations**" in report else "")
                ]
                
                for title, body in sections:
                    if body.strip():
                        pdf.add_section(title, body)
                
                image = st.session_state.report_data["image"]
                predicted_idx = classes.index(st.session_state.report_data["prediction"])
                pdf.add_explainability(
                    apply_lime(image, model_cnn, classes),
                    apply_integrated_gradients(image, model_cnn, predicted_idx),
                    apply_gradcam(image, model_cnn, predicted_idx)
                )
                
                pdf_output = pdf.output(dest="S").encode('latin1', 'ignore')
                b64 = base64.b64encode(pdf_output).decode('latin1')
                
                st.markdown(f"""
                <a href="data:application/pdf;base64,{b64}" download="NutriScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf" class="download-btn">
                    📥 Download Your Report
                </a>
                """, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
else:
    st.info("Run an analysis to generate a report.")

# Line 731: Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ by <b>Ujjwal Sinha</b> • "
    "<a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)

# Line 735: Final memory cleanup
torch.mps.empty_cache()