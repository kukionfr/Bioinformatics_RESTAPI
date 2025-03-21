import re
import cv2
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import uuid
from geojson import Feature, FeatureCollection
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi.responses import JSONResponse, FileResponse
from shapely.geometry import Polygon, shape
import json
import matplotlib
matplotlib.use('Agg')


app = FastAPI()

SAVE_DIR = "./asset/processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)
# Global dictionary to store KMeans results
kmeans_results = {}

class DNASequence(BaseModel):
    sequence: str

def validate_dna(seq: str):
    if not re.match("^[ACGTacgt]+$", seq):
        raise HTTPException(status_code=400, detail="Invalid DNA sequence. Only A, C, G, T are allowed.")

def gc_content(seq: str) -> float:
    gc_count = sum(1 for base in seq if base in "GCgc")
    return round((gc_count / len(seq)) * 100, 2) if seq else 0.0

def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def transcribe(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")

def process_image(image_bytes: bytes, filename: str):
    """ Process microscopic image using refined DAPI-stained nuclei detection and save the result. """
    # Convert bytes to OpenCV image
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format

    # Convert to HSV and LAB color spaces for better blue detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Create masks based on optimized color thresholds
    mask_hsv = cv2.inRange(hsv, np.array([60, 50, 50]), np.array([87, 255, 255]))
    mask_lab = cv2.inRange(lab, np.array([128, 127, 128]), np.array([200, 128, 128]))

    # Combine masks to enhance detection
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    _, binary_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(~binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500  # Minimum area threshold to keep objects
    circularity_threshold = 0.5  # Circularity threshold to detect circular objects

    # List to store circular objects that are big enough
    circular_objects = []
    nuclei_shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0 or area < min_area:  # Skip very small objects
            continue

        # Compute circularity: 4π * Area / Perimeter^2
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Keep only objects that are circular and large enough
        if circularity >= circularity_threshold:
            circular_objects.append(contour)
            nuclei_shapes.append([round(area,2),round(perimeter,2),round(circularity,2)])
    csv_filename = f"{filename.split('.')[0]}.csv"
    pd.DataFrame(nuclei_shapes, columns=['area', 'perimeter', 'circularity']).to_csv(os.path.join(SAVE_DIR, csv_filename), index=False)

    num_cells = len(circular_objects)

    # Draw detected nuclei on the original image
    image_output = image.copy()
    cv2.drawContours(image_output, circular_objects, -1, (0, 0, 255), 3)  # Green contours for nuclei

    # Save the processed image
    processed_filename = os.path.join(SAVE_DIR, f"{filename.split('.')[0]}.png")
    cv2.imwrite(processed_filename, image_output)

    # output contour
    # Convert contours into a DataFrame
    df = pd.DataFrame({'contours': [Polygon(contour[:, 0, :]) for contour in circular_objects]})

    # Create GeoJSON features
    df['feat'] = df.apply(lambda x: Feature(
        geometry=x['contours'],
        properties={"objectType": "annotation"},
        id=str(uuid.uuid1())
    ), axis=1)

    # Convert to GeoJSON structure
    geojson = FeatureCollection(df['feat'].tolist())

    # Save to file
    geojson_path = f"{SAVE_DIR}/{filename.split('.')[0]}.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=4)

    return processed_filename, csv_filename, geojson_path, num_cells

@app.post("/gc-content/")
async def get_gc_content(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "gc_content": gc_content(data.sequence)}

@app.post("/reverse-complement/")
async def get_reverse_complement(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "reverse_complement": reverse_complement(data.sequence)}

@app.post("/transcribe/")
async def get_transcription(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "transcription": transcribe(data.sequence)}

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    """API to analyze image and return download link."""

    image_bytes = await file.read()
    processed_filepath, csv_filename, geojson_path, num_cells = process_image(image_bytes, file.filename)

    return {
        "filename": file.filename,
        "num_cells_detected": num_cells,
        "processed_image_url": f"http://127.0.0.1:8000/download/{os.path.basename(processed_filepath)}",
        "nuclei_shape_dataframe": f"http://127.0.0.1:8000/download/{os.path.basename(csv_filename)}",
        "nuclei_contour": f"http://127.0.0.1:8000/download/{os.path.basename(geojson_path)}"
    }


@app.get("/download/{filename}")
async def download_processed_image(filename: str):
    """ Endpoint to serve processed images. """
    file_path = os.path.join(SAVE_DIR, filename)
    return FileResponse(file_path, media_type="image/png", filename=filename)

@app.get("/eda")
def exploratory_data_analysis(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")
    df = pd.read_csv(csv_path)
    summary = df.describe().to_dict()
    return JSONResponse(content=summary)


@app.get("/pairplot")
def generate_pairplot(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 6))
    sns.pairplot(df)
    pairplot_path = os.path.join(SAVE_DIR, "pairplot.png")
    plt.savefig(pairplot_path, format='png')  # Ensure correct format
    plt.close()
    return FileResponse(pairplot_path, media_type="image/png")


@app.get("/kmeans")
def kmeans_clustering(csv_path: str, n_clusters: int = 3):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")
    df = pd.read_csv(csv_path)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df)
    df['cluster'] = clusters
    # Store results globally
    kmeans_results[csv_path] = clusters.tolist()
    cluster_centers = kmeans.cluster_centers_.tolist()

    # Overwrite the original CSV with the new column
    df.to_csv(csv_path, index=False)

    geojson_path = csv_path.replace('csv','geojson')

    # Load the GeoJSON file
    with open(geojson_path, "r") as f:
        loaded_geojson = json.load(f)

    # Convert GeoJSON back to list of contours (NumPy arrays)
    contours_list = [
        np.array(shape(feature["geometry"]).exterior.coords, dtype=np.int32)
        for feature in loaded_geojson["features"]
    ]
    print(len(contours_list))

    image_path = csv_path.replace('csv','png')
    image = cv2.imread(image_path)
    # Loop through contours
    for idx,cnt in enumerate(contours_list):
        M = cv2.moments(cnt)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cluster = df['cluster'].iloc[idx]
            # draw.text((cx, cy), str(cluster), fill="red", font=font)
            cv2.putText(image, str(cluster), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
        else:
            print("Zero contour area, no centroid.")

    # Save the clustered image
    processed_filename = image_path.replace('.png', '_clustered.png')
    # Save
    cv2.imwrite(processed_filename, image)
    # image.save(processed_filename)
    return {"clusters": df['cluster'].tolist(), "centroids": cluster_centers,
            "message": "CSV file updated with cluster column",
            "processed_image_url": f"http://127.0.0.1:8000/download/{os.path.basename(processed_filename)}",
            }


# @app.get("/pca")
# def pca_analysis(csv_path: str):
#     if not os.path.exists(csv_path):
#         raise HTTPException(status_code=400, detail="CSV file not found.")
#     df = pd.read_csv(csv_path)
#
#     pca = PCA(n_components=2)
#     principal_components = pca.fit_transform(df)
#     df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
#
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'])
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     pca_path = os.path.join(SAVE_DIR, "pca_plot.png")
#     plt.savefig(pca_path, format='png')  # Ensure correct format
#     plt.close()
#     return FileResponse(pca_path, media_type="image/png")
@app.get("/pca")
def pca_analysis(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")

    df = pd.read_csv(csv_path)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

    # Retrieve cluster labels if KMeans was run before
    clusters = kmeans_results.get(csv_path, [None] * len(df_pca))  # Default to None if missing

    df_pca['Cluster'] = clusters

    # Plot PCA with cluster labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Cluster'], palette="tab10")

    # Annotate points with cluster numbers
    for i, (x, y) in enumerate(zip(df_pca['PC1'], df_pca['PC2'])):
        plt.text(x, y, str(df_pca['Cluster'][i]), fontsize=9, ha='right', color='black')

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Plot with Cluster Labels")
    plt.legend(title="Clusters")

    # Save the plot
    pca_path = os.path.join(SAVE_DIR, "pca_plot.png")
    plt.savefig(pca_path, format='png')
    plt.close()

    return FileResponse(pca_path, media_type="image/png")

@app.post("/upload_csv")
def upload_csv(file: UploadFile = File(...)):
    try:
        file_location = f"./asset/processed_images/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())
        return {"message": "CSV uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Bioinformatics REST API"}



