from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import re
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
from fastapi.responses import FileResponse
import pandas as pd

app = FastAPI()

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


SAVE_DIR = "./asset/processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)

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
            nuclei_shapes.append([area,perimeter,circularity])
    csv_filename = f"{filename.split('.')[0]}nuclei_shape_features.csv"
    pd.DataFrame(nuclei_shapes, columns=['area', 'perimeter', 'circularity']).to_csv(os.path.join(SAVE_DIR, csv_filename), index=False)

    num_cells = len(circular_objects)

    # Draw detected nuclei on the original image
    image_output = image.copy()
    cv2.drawContours(image_output, circular_objects, -1, (0, 0, 255), 3)  # Green contours for nuclei

    # Save the processed image
    processed_filename = os.path.join(SAVE_DIR, f"processed_{filename.split('.')[0]}.png")
    cv2.imwrite(processed_filename, image_output)

    return processed_filename, csv_filename, num_cells

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
    processed_filepath, csv_filename, num_cells = process_image(image_bytes, file.filename)

    return {
        "filename": file.filename,
        "num_cells_detected": num_cells,
        "processed_image_url": f"http://127.0.0.1:8000/download/{os.path.basename(processed_filepath)}",
        "nuclei_shape_dataframe": f"http://127.0.0.1:8000/download/{os.path.basename(csv_filename)}"
    }


@app.get("/download/{filename}")
async def download_processed_image(filename: str):
    """ Endpoint to serve processed images. """
    file_path = os.path.join(SAVE_DIR, filename)
    return FileResponse(file_path, media_type="image/png", filename=filename)
@app.get("/")
async def root():
    return {"message": "Welcome to the Bioinformatics REST API"}



