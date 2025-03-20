from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Directory for saving processed images
SAVE_DIR = "./asset/processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)


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
    plt.savefig(pairplot_path)
    plt.close()
    return FileResponse(pairplot_path, media_type="image/png")


@app.get("/kmeans")
def kmeans_clustering(csv_path: str, n_clusters: int = 3):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")
    df = pd.read_csv(csv_path)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df)
    cluster_centers = kmeans.cluster_centers_.tolist()
    return {"clusters": df['cluster'].tolist(), "centroids": cluster_centers}


@app.get("/pca")
def pca_analysis(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found.")
    df = pd.read_csv(csv_path)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    pca_path = os.path.join(SAVE_DIR, "pca_plot.png")
    plt.savefig(pca_path)
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
