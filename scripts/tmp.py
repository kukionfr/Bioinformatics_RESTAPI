from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import os
import urllib.request
import gzip
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Create directory for output
output_dir = "./asset/gwas_results"
os.makedirs(output_dir, exist_ok=True)

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
    filename = os.path.join(output_dir, os.path.basename(url))
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

    result_csv = os.path.join(output_dir, "gwas_results.csv")
    results.to_csv(result_csv, index=False)

    # Also generate plots
    manhattan_path = os.path.join(output_dir, "manhattan_plot.png")
    qq_path = os.path.join(output_dir, "qq_plot.png")
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
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=filename)