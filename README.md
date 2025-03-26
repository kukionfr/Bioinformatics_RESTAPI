# 🧬 Bioinformatics REST API

## 🚀 Overview
Welcome to the **Bioinformatics REST API**, a FastAPI-powered service for:

- 🔬 **DNA Sequence Analysis**
- 🧠 **Microscopic Nuclei Segmentation**
- 🤖 **Machine Learning on Nuclei Shape Features**

This API enables bioinformatics workflows with just a few HTTP calls — from GC content analysis to K-means clustering of nuclei shapes.

---

## 🐳 Setup: Docker Environment

### 🔧 Build the Docker Image
```bash
docker build -t bioinformatics_restapi .
```

### 🚦 Run the Container
```bash
docker run -d -p 8000:8000 bioinformatics_restapi
```
### 🔍 Check if it's running
```bash
curl -X GET "http://127.0.0.1:8000/"
```
**Expected:**
```json
{"message":"Welcome to the Bioinformatics REST API"}
```

---

## 🧪 DNA Sequence Analysis
### 🔬 Endpoints

#### 1️⃣ GC Content
```bash
curl -X POST "http://127.0.0.1:8000/gc-content/" -H "Content-Type: application/json" -d '{"sequence": "ATGCGC"}'
```
#### 1️⃣ GC Content for windows 
```
curl -X POST "http://127.0.0.1:8000/gc-content/" -H "Content-Type: application/json" -d "{\"sequence\": \"ATGCGC\"}"
```
> **Response:**
```json
{"sequence":"ATGCGC","gc_content":66.67}
```

#### 2️⃣ Reverse Complement
```bash
curl -X POST "http://127.0.0.1:8000/reverse-complement/" -H "Content-Type: application/json" -d '{"sequence": "ATGCGC"}'
```
> **Response:**
```json
{"sequence":"ATGCGC","reverse_complement":"GCGCAT"}
```

#### 3️⃣ Transcription (DNA ➜ RNA)
```bash
curl -X POST "http://127.0.0.1:8000/transcribe/" -H "Content-Type: application/json" -d '{"sequence": "ATGCGC"}'
```
> **Response:**
```json
{"sequence":"ATGCGC","transcription":"AUGCGC"}
```

---

## 🧫 Nuclei Segmentation (Image Analysis)

### 📤 Upload Image
```bash
curl -X POST "http://127.0.0.1:8000/analyze-image/"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@./asset/microscopic_sample.jpg"
```

📷 Sample image:
![Microscopic Sample](./asset/readme_asset/microscopic_sample.jpg)

> **Response:**
```json
{
  "filename": "microscopic_sample.jpg",
  "num_cells_detected": 19,
  "processed_image_url": "http://127.0.0.1:8000/download/microscopic_sample.png",
  "nuclei_shape_dataframe": "http://127.0.0.1:8000/download/microscopic_sample.csv",
  "nuclei_contour": "http://127.0.0.1:8000/download/microscopic_sample.geojson"
}
```

📌 Outputs:
- ![Processed Image](./asset/readme_asset/microscopic_sample.png)
- [📄 Nuclei Shape CSV](./asset/readme_asset/microscopic_sample.csv)
- [🧬 GeoJSON for QuPath](http://127.0.0.1:8000/download/microscopic_sample.geojson)

🔎 View in [QuPath](https://qupath.github.io/):
![QuPath Screenshot](./asset/readme_asset/qupath_screenshot.png)

---

## 🧠 Machine Learning on Nuclei Shape

### 📊 1. Exploratory Data Analysis (EDA)
```bash
curl -X GET "http://127.0.0.1:8000/eda?csv_path=./asset/processed_images/microscopic_sample.csv"
```
> **Response:**
Returns summary statistics (mean, std, percentiles) in JSON format.

---

### 🔁 2. Pairplot Visualization
```bash
curl -X GET "http://127.0.0.1:8000/pairplot?csv_path=./asset/processed_images/microscopic_sample.csv" --output "./asset/pairplot.png"
```
📷 Output:
![Pairplot](./asset/readme_asset/pairplot.png)

---

### 🔀 3. K-Means Clustering
```bash
curl -X GET "http://127.0.0.1:8000/kmeans?csv_path=./asset/processed_images/microscopic_sample.csv"
```

🎯 Output:
![Clustered Image](./asset/readme_asset/microscopic_sample_clustered.png)

🧩 Download the image directly:
[http://127.0.0.1:8000/download/microscopic_sample_clustered.png](http://127.0.0.1:8000/download/microscopic_sample_clustered.png)

---

### 🧭 4. PCA Plot (2D Visualization)
```bash
curl -X GET "http://127.0.0.1:8000/pca?csv_path=./asset/processed_images/microscopic_sample.csv" --output "./asset/pca_plot.png"
```

🖼️ Output:
![PCA Plot](./asset/readme_asset/pca_plot.png)

---

## 🤝 Contributions & Support

Found a bug? Want to add a feature?

- 🌱 Fork the repo
- 🛠️ Open a PR
- 🐛 Or file an issue

Thanks for stopping by! 💙
