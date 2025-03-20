# Bioinformatics REST API

## Overview
This repository provides a REST API for DNA sequence analysis and nuclei segmentation. The API is implemented using FastAPI and includes functionality for computing GC content, generating reverse complements, transcribing DNA to RNA, and analyzing microscopic images for nuclei segmentation.

---

## **Setting Up the Environment**

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
curl -X POST "http://127.0.0.1:8000/gc-content/" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","gc_content":66.67}
```

#### **2. Get Reverse Complement**
```sh
curl -X POST "http://127.0.0.1:8000/reverse-complement/" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","reverse_complement":"GCGCAT"}
```

#### **3. Transcribe DNA to RNA**
```sh
curl -X POST "http://127.0.0.1:8000/transcribe/" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATGCGC"}'
```
##### **Expected Response:**
```json
{"sequence":"ATGCGC","transcription":"AUGCGC"}
```

---

## **Nuclei Segmentation**


### **Uploading an Image for Analysis**
```sh
curl -X POST "http://127.0.0.1:8000/analyze-image/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./asset/microscopic_sample.jpg"
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

## **Contributions & Support**
For any issues or contributions, please open a pull request or issue on this repository. 

