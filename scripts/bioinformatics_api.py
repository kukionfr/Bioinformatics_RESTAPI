import re
import cv2
from io import BytesIO
from PIL import Image
import uuid
from geojson import Feature, FeatureCollection
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from shapely.geometry import Polygon, shape
import json
import matplotlib
matplotlib.use('Agg')
import urllib.request
import gzip
import statsmodels.api as sm
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

app = FastAPI()

# RDF Setup
BIO = Namespace("http://example.org/bio#")
SEQUENCE_BASE = "http://example.org/sequence/"
graph = Graph()

SAVE_DIR = "./asset/generated_output"
os.makedirs(SAVE_DIR, exist_ok=True)
# Global dictionary to store KMeans results
kmeans_results = {}

class DNASequence(BaseModel):
    sequence: str

class SPARQLQuery(BaseModel):
    query: str

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


# ----------- Utility Functions -----------
def download_file(url: str, destination: str):
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def convert_genotype(gt):
    if gt in {'0|0', '0/0'}:
        return 0
    elif gt in {'0|1', '1|0', '0/1', '1/0'}:
        return 1
    elif gt in {'1|1', '1/1'}:
        return 2
    else:
        return np.nan

def parse_vcf_subset(vcf_path, max_snps=200):
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                header = line.strip().split('\t')
                samples = header[9:]
                break

        genotype_data = {}
        snp_count = 0

        for line in f:
            if line.startswith('#'):
                continue
            if snp_count >= max_snps:
                break
            parts = line.strip().split('\t')
            snp_id = parts[2] if parts[2] != '.' else f"{parts[0]}:{parts[1]}"
            genotypes = [convert_genotype(g.split(':')[0]) for g in parts[9:]]
            genotype_data[snp_id] = genotypes
            snp_count += 1

    genotype_df = pd.DataFrame(genotype_data, index=samples)
    genotype_df.dropna(inplace=True)
    return genotype_df

def simulate_covariates_and_phenotype(df):
    np.random.seed(42)
    df['Gender'] = np.random.choice([0, 1], size=len(df))
    df['Population'] = np.random.choice([0, 1, 2], size=len(df))
    pcs = PCA(n_components=2).fit_transform(df.drop(columns=['Gender', 'Population']))
    df['PC1'], df['PC2'] = pcs[:, 0], pcs[:, 1]
    df['Phenotype'] = (
        0.4 * df['Gender'] +
        0.5 * df['Population'] +
        0.3 * df['PC1'] -
        0.2 * df['PC2'] +
        np.random.normal(0, 1, size=len(df))
    )
    return df

def run_multivariate_gwas(df, covariates):
    results = []
    for snp in df.columns.difference(covariates + ['Phenotype']):
        X = df[[snp] + covariates]
        X = sm.add_constant(X)
        y = df['Phenotype']
        try:
            model = sm.OLS(y, X).fit()
            results.append({
                'SNP': snp,
                'beta': model.params[snp],
                'pval': model.pvalues[snp]
            })
        except Exception:
            continue
    results_df = pd.DataFrame(results)
    results_df['-log10(p)'] = -np.log10(results_df['pval'])
    results_df.sort_values('pval', inplace=True)
    return results_df

def plot_manhattan(df, out_path):
    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(df)), df['-log10(p)'], s=10)
    plt.axhline(-np.log10(5e-8), color='red', linestyle='--', label='Genome-wide significance')
    plt.xlabel("SNP Index")
    plt.ylabel("-log10(p-value)")
    plt.title("Manhattan Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_qq(df, out_path):
    observed = -np.log10(np.sort(df['pval'].dropna()))
    expected = -np.log10(np.linspace(1 / len(observed), 1, len(observed)))
    plt.figure(figsize=(6, 6))
    plt.plot(expected, observed, 'o', markersize=3, label='Observed')
    plt.plot([0, max(expected)], [0, max(expected)], 'r--', label='Expected')
    plt.xlabel('Expected -log10(p)')
    plt.ylabel('Observed -log10(p)')
    plt.title('QQ Plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ----------- API Endpoints -----------

@app.get("/download-vcf")
def download_vcf_from_aws(url: str = Query(...)):
    filename = os.path.join(SAVE_DIR, os.path.basename(url))
    download_file(url, filename)
    return {"message": "VCF file downloaded", "path": filename}

@app.get("/run-gwas")
def run_gwas(vcf_path: str, snps: int = 200):
    if not os.path.exists(vcf_path):
        raise HTTPException(status_code=400, detail="VCF file not found.")

    df = parse_vcf_subset(vcf_path, max_snps=snps)
    df = simulate_covariates_and_phenotype(df)
    covars = ['Gender', 'Population', 'PC1', 'PC2']
    results = run_multivariate_gwas(df, covars)

    result_csv = os.path.join(SAVE_DIR, "gwas_results.csv")
    results.to_csv(result_csv, index=False)

    # Also generate plots
    manhattan_path = os.path.join(SAVE_DIR, "manhattan_plot.png")
    qq_path = os.path.join(SAVE_DIR, "qq_plot.png")
    plot_manhattan(results, manhattan_path)
    plot_qq(results, qq_path)

    return {
        "message": "GWAS completed",
        "num_snps_tested": len(results),
        "result_file": result_csv,
        "manhattan_plot": manhattan_path,
        "qq_plot": qq_path
    }

@app.get("/download/{filename}")
def download_file_endpoint(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=filename)

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
            continue

    # Save the clustered image
    # processed_filename = os.path.join(SAVE_DIR, "microscopic_sample_clustered.png")
    processed_filename = image_path.replace('.png', '_clustered.png')
    # Save
    cv2.imwrite(processed_filename, image)

    # Optional: Force sync
    # with open(processed_filename, 'rb') as f:
    #     os.fsync(f.fileno())

    if not os.path.exists(processed_filename):
        raise HTTPException(status_code=500, detail="Processed image not found.")

    if os.path.getsize(processed_filename) == 0:
        raise HTTPException(status_code=500, detail="Processed image is empty.")

    return FileResponse(processed_filename, media_type="image/png", filename="microscopic_sample_clustered.png")

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

@app.post("/sequence/rdf", response_class=PlainTextResponse)
def add_sequence_rdf(dna: DNASequence):
    validate_dna(dna.sequence)
    seq_id = str(uuid.uuid4())
    seq_uri = URIRef(SEQUENCE_BASE + seq_id)

    graph.add((seq_uri, RDF.type, BIO.DNASequence))
    graph.add((seq_uri, BIO.sequence, Literal(dna.sequence)))
    graph.add((seq_uri, BIO.gcContent, Literal(gc_content(dna.sequence), datatype=XSD.float)))
    graph.add((seq_uri, BIO.reverseComplement, Literal(reverse_complement(dna.sequence))))
    graph.add((seq_uri, BIO.transcript, Literal(transcribe(dna.sequence))))

    return graph.serialize(format="turtle")

@app.get("/sparql")
def sparql_query_get(query: str = Query(..., description="SPARQL query string")):
    try:
        result = graph.query(query)
        bindings = []
        for row in result:
            bindings.append({f"var{i}": str(val) for i, val in enumerate(row)})
        return JSONResponse(content={"results": bindings})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sparql")
def sparql_query_post(sparql: SPARQLQuery):
    try:
        result = graph.query(sparql.query)
        bindings = []
        for row in result:
            bindings.append({f"var{i}": str(val) for i, val in enumerate(row)})
        return JSONResponse(content={"results": bindings})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Example RDF namespace setup
graph.bind("bio", BIO)
@app.get("/")
async def root():
    return {"message": "Welcome to the Bioinformatics REST API"}



