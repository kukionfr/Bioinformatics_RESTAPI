# Bioinformatics REST API

## Overview
This repository provides a REST API for DNA sequence analysis, nuclei segmentation, and machine learning-based analysis of microscopic nuclei shape features. The API is implemented using FastAPI and includes functionality for computing GC content, generating reverse complements, transcribing DNA to RNA, analyzing microscopic images for nuclei segmentation, and performing machine learning analysis on nuclei shape data.

---

## **Setting Up the Environment**

Clone this repository in your method of preference

To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

Ensure all dependencies are correctly installed before running the API.

---

## **DNA Sequence Analysis**

### **Starting the API**
Navigate to the `scripts` directory and run the FastAPI server:
```sh
uvicorn scripts.bioinformatics_api:app --host 0.0.0.0 --port 8000 --reload
```

### **Checking if the API is Running**
Open a new terminal and execute:
```sh
curl -X GET "http://127.0.0.1:8000/"
```
#### **Expected Response:**
```json
{"message":"Welcome to the Bioinformatics REST API"}
```

### **Available Endpoints**

#### **1. Calculate GC Content**
```sh
curl -X POST "http://127.0.0.1:8000/gc-content/"      -H "Content-Type: application/json"      -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","gc_content":66.67}
```

#### **2. Get Reverse Complement**
```sh
curl -X POST "http://127.0.0.1:8000/reverse-complement/"      -H "Content-Type: application/json"      -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","reverse_complement":"GCGCAT"}
```

#### **3. Transcribe DNA to RNA**
```sh
curl -X POST "http://127.0.0.1:8000/transcribe/"      -H "Content-Type: application/json"      -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","transcription":"AUGCGC"}
```

---

## **Nuclei Segmentation**

### **Uploading an Image for Analysis**
```sh
curl -X POST "http://127.0.0.1:8000/analyze-image/"      -H "accept: application/json"      -H "Content-Type: multipart/form-data"      -F "file=@./asset/microscopic_sample.jpg"
```
##### **Expected Response:**
```json
{"filename":"microscopic_sample.jpg",
  "num_cells_detected":19,
  "processed_image_url":"http://127.0.0.1:8000/download/processed_microscopic_sample.png",
  "nuclei_shape_dataframe":"http://127.0.0.1:8000/download/microscopic_samplenuclei_shape_features.csv"}
```

##### **Click the processed_image_url to download the segmented nuclei image:**
![Microscopic Sample](./asset/processed_images/processed_microscopic_sample.png)

[View Nuclei Shape Features](./asset/processed_images/microscopic_samplenuclei_shape_features.csv)

---

## **Machine Learning Analysis on Nuclei Shape Features**

### **1. Exploratory Data Analysis (EDA)**
```sh
curl -X GET "http://127.0.0.1:8000/eda?csv_path=./asset/processed_images/microscopic_samplenuclei_shape_features.csv"
```
##### **Expected Response:**
Returns JSON-formatted summary statistics of the dataset.

### **2. Pairplot Visualizats
```sh
curl -X GET "http://127.0.0.1:8000/pairplot?csv_path=./asset/processed_images/microscopic_samplenuclei_shape_features.csv"
```
##### **Expected Response:**
Returns an image file showing pairwise relationships in the dataset.

### **3. K-Means Clustering**
```sh
curl -X GET "http://127.0.0.1:8000/kmeans?n_clusters=3"
```
##### **Expected Response:**
```json
{"clusters": [...], "centroids": [...]}
```
Returns cluster assignments for each data point and cluster centroids.

### **4. Principal Component Analysis (PCA) Visualization**
```sh
curl -X GET "http://127.0.0.1:8000/pca"
```
##### **Expected Response:**
Returns an image file showing a 2D visualization of the dataset after PCA dimensionality reduction.

### **5. Upload a New CSV File for Analysis**
```sh
curl -X POST "http://127.0.0.1:8000/upload_csv"      -H "accept: application/json"      -H "Content-Type: multipart/form-data"      -F "file=@./asset/processed_images/new_dataset.csv"
```
##### **Expected Response:**
```json
{"message": "CSV uploaded successfully", "filename": "new_dataset.csv"}
```
Allows users to upload a new dataset for analysis.

---

## **Contributions & Support**
For any issues or contributions, please open a pull request or issue on this repository.
