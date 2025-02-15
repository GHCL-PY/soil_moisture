from fastapi import FastAPI, UploadFile, File, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import ImageAnalysis
import shutil
import os
import zipfile
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


app = FastAPI()
Base.metadata.create_all(bind=engine)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def analys_image(image_path):
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Load the satellite image
    # image_path = "data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_050.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to LAB color space and apply median blur
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lab = cv2.medianBlur(image_lab, 3)  # Reduce noise
    lab_flattened = image_lab.reshape((-1, 3))

    # K-Means Clustering
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(lab_flattened)
    cluster_labels = kmeans.labels_.reshape(image_lab.shape[:2])

    # Analyze cluster LAB means
    cluster_means = kmeans.cluster_centers_
    for i, mean in enumerate(cluster_means):
        print(f"Cluster {i}: LAB mean = {mean}")

    # Dynamic Mapping of Clusters
    sorted_indices = np.argsort([mean[1] for mean in cluster_means])  # Sort by LAB brightness
    category_mapping = {
        sorted_indices[0]: 0,  # Water
        sorted_indices[1]: 1,  # Dry Soil
        sorted_indices[2]: 2,  # Wet Soil
        sorted_indices[3]: 3,  # Salty-Muddy
    }
    classified_image = np.vectorize(category_mapping.get)(cluster_labels)

    # Calculate Percentages
    total_pixels = classified_image.size
    percentages = {
        'Water': np.sum(classified_image == 0) / total_pixels * 100,
        'Wet Soil': np.sum(classified_image == 1) / total_pixels * 100,
        'Salty Mud': np.sum(classified_image == 2) / total_pixels * 100,
        'Dry Soil': np.sum(classified_image == 3) / total_pixels * 100,
    }
    return percentages


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("index.html", {"request": request, "selected_images": None, "plot_url": "/static/plot.png"})
import re
def extract_date(image_path):
    # Set the path to tesseract.exe (change if installed elsewhere)

    # image_path = "D:/GHCLDATA/soil_moisture_analysis/data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_030.jpg"

    # Load the image
    image = cv2.imread(image_path)  # Replace with your image file

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = gray.shape

    # Crop the top-right corner (adjust values if needed)
    x1, y1 = int(width * 0.75), 0   # 75% width from the left
    x2, y2 = width, int(height * 0.2)  # Top 20% of the image
    cropped = gray[y1:y2, x1:x2]

    # Apply thresholding for better OCR
    _, thresh = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(thresh, config="--psm 6")
    ocr_date = extracted_text.strip()

    # Print extracted text
    print("Extracted Date:", ocr_date)
    try:
        match = re.search(r"\d{4}-\d{2}-\d{2}", ocr_date)
        if match:
            extracted_date = match.group(0)
            print("++++++++++++++++++++++", extracted_date)

            date_obj = datetime.datetime.strptime(extracted_date, "%Y-%m-%d").date()  # Ensure only date
            print(date_obj,"$$$$$$$$$$$")
            return date_obj
    except:
        pass  # Move to filename extraction if error

    # Extract YYYY-MM-DD from filename
    image_filename = os.path.basename(image_path)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", image_filename)
    if match:
        return datetime.datetime.strptime(match.group(0), "%Y-%m-%d").date()  # Convert to date

    # incase you need to remove current date then need to change image name (add date formate yyyy/mm/dd)
    return datetime.datetime.today().date()


@app.post("/upload/")
async def upload_images(request: Request,files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    selected_images = []

    for file in files:
        if file.filename.endswith(".zip"):
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
            os.remove(zip_path)
            extracted_files = os.listdir(UPLOAD_FOLDER)
        else:
            extracted_files = [file.filename]
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        for img in extracted_files:
            image_path = f"uploads/{img}"
            image_date = extract_date(image_path)
            print(image_date,"===================")
            # if not image_date:
            #     image_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            process_date = datetime.datetime.utcnow()
            unique_key = int(process_date.timestamp())

            #check db existing entry
            existing_entry = db.query(ImageAnalysis).filter(
                (ImageAnalysis.image_name == img) | (ImageAnalysis.unique_key == unique_key)
            ).first()

            if existing_entry:
                selected_images.append(existing_entry)
                continue  
            analysis = analys_image(image_path) 

            # analysis = {
            #     "Water": np.random.uniform(10, 70),
            #     "Dry Soil": np.random.uniform(10, 40),
            #     "Wet Soil": np.random.uniform(5, 30),
            #     "Salty Mud": np.random.uniform(1, 25)
            # }np.float64(24.63)

            new_entry = ImageAnalysis(
                image_name=img,
                image_date=image_date,
                process_date=process_date,
                water_percent=round(float(analysis["Water"]),2),
                dry_soil_percent=round(float(analysis["Dry Soil"]),2),
                wet_soil_percent=round(float(analysis["Wet Soil"]),2),
                salty_muddy_percent=round(float(analysis["Salty Mud"]),2),
                unique_key=unique_key
            )

            db.add(new_entry)
            db.commit()
            selected_images.append(new_entry)
    # generate_plots()


    return templates.TemplateResponse("index.html", {"request": request, "selected_images": selected_images,"plot_generated": True, "plot_url": "/static/plot.png"})


import seaborn as sns



# Create 'static/plots/' directory if not exists
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

@app.get("/plots/")
def generate_plots():
    """Generate KDE + Bar plots for each parameter and save them as PNG images."""
    db = next(get_db())
    
    data = db.query(ImageAnalysis).all()
    if not data:
        return {"message": "No data available"}

    df = pd.DataFrame([{
        "month": datetime.datetime.strptime(d.image_date, "%Y-%m-%d").month,
        "year": datetime.datetime.strptime(d.image_date, "%Y-%m-%d").year,
        "water": d.water_percent,
        "dry_soil": d.dry_soil_percent,
        "wet_soil": d.wet_soil_percent,
        "salty_muddy": d.salty_muddy_percent
    } for d in data])

    # Group by month and year, calculating mean, min, max
    grouped = df.groupby(["year", "month"]).agg({
        "water": ["mean", "min", "max"],
        "dry_soil": ["mean", "min", "max"],
        "wet_soil": ["mean", "min", "max"],
        "salty_muddy": ["mean", "min", "max"]
    }).reset_index()

    # Rename columns
    grouped.columns = ["year", "month",
                       "water_avg", "water_min", "water_max",
                       "dry_avg", "dry_min", "dry_max",
                       "wet_avg", "wet_min", "wet_max",
                       "salty_avg", "salty_min", "salty_max"]

    # Define parameter names
    params = ["water", "dry_soil", "wet_soil", "salty_muddy"]
    titles = ["Water %", "Dry Soil %", "Wet Soil %", "Salty Mud %"]
    colors = ["blue", "brown", "green", "purple"]

    param_map = {
        "water": "water",
        "dry_soil": "dry",
        "wet_soil": "wet",
        "salty_muddy": "salty"
    }

    # Ensure we group by month only to have 12 values
    grouped = df.groupby("month").agg({
        "water": ["mean", "min", "max"],
        "dry_soil": ["mean", "min", "max"],
        "wet_soil": ["mean", "min", "max"],
        "salty_muddy": ["mean", "min", "max"]
    }).reset_index()

    # Rename columns for easy access
    grouped.columns = ["month",
                    "water_avg", "water_min", "water_max",
                    "dry_avg", "dry_min", "dry_max",
                    "wet_avg", "wet_min", "wet_max",
                    "salty_avg", "salty_min", "salty_max"]

    fig, axes = plt.subplots(2, 2, figsize=(6, 4), constrained_layout=True)
    axes = axes.flatten()


    for i, (param, color) in enumerate(zip(["water", "dry_soil", "wet_soil", "salty_muddy"], ["blue", "brown", "green", "purple"])):
        ax = axes[i]
        
        mapped_param = param_map[param]  # Map to correct column name
        
        # Bar Plot
        ax.bar(grouped["month"], grouped[f"{mapped_param}_avg"], color=color, alpha=0.4, label="Avg")

        # Min-Max Markers
        ax.scatter(grouped["month"], grouped[f"{mapped_param}_min"], color="red", label="Min",s = 3, marker="o")
        ax.scatter(grouped["month"], grouped[f"{mapped_param}_max"], color="green", label="Max", s = 3, marker="o")

        ax.set_title(param.capitalize(), fontsize=6, fontweight="bold", color=color)
        ax.set_xlabel("Month", fontsize=4)
        ax.set_ylabel("Percentage", fontsize=4)
        ax.legend(fontsize=3,markerscale=0.4)
        ax.set_xticks(range(1, 13))  # Ensure only 12 months are displayed
        ax.tick_params(axis='x', labelsize=5)  # Set x-axis tick size
        ax.tick_params(axis='y', labelsize=5)  # Set y-axis tick size
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plot_path = os.path.join(PLOT_DIR, "soil_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return FileResponse(plot_path, media_type="image/png")

    


