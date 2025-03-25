# 📊 GWAS FastAPI

A simple FastAPI service for running multivariate GWAS (Genome-Wide Association Studies) using simulated phenotype and covariates on real VCF data.

---

## 🚀 Setup

Install dependencies:

```bash
pip install fastapi uvicorn pandas numpy statsmodels scikit-learn matplotlib
```

Run the API:

```bash
uvicorn gwas_fastapi_app:app --reload
```

---

## 🧪 Endpoints

### 1. **Download VCF from AWS**

Download a VCF file from a public URL (e.g., AWS 1000 Genomes):

```bash
curl -X GET "http://127.0.0.1:8000/download-vcf?url=https://1000genomes.s3.amazonaws.com/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
```

**Response:**
```json
{
  "message": "VCF file downloaded",
  "path": "./asset/gwas_results/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
}
```

---

### 2. **Run GWAS**

Run a multivariate GWAS on a local VCF file with a specified number of SNPs:

```bash
curl -X GET "http://127.0.0.1:8000/run-gwas?vcf_path=./asset/gwas_results/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz&snps=200"
```

**Response:**
```json
{
  "message": "GWAS completed",
  "num_snps_tested": 200,
  "result_file": "./asset/gwas_results/gwas_results.csv",
  "manhattan_plot": "./asset/gwas_results/manhattan_plot.png",
  "qq_plot": "./asset/gwas_results/qq_plot.png"
}
```

---

### 3. **Download Output Files**

Download result files (CSV or plots):

```bash
curl -O http://127.0.0.1:8000/download/gwas_results.csv
curl -O http://127.0.0.1:8000/download/manhattan_plot.png
curl -O http://127.0.0.1:8000/download/qq_plot.png
```

---

## 📝 Notes
- Phenotype is simulated as a linear combination of gender, population, and genotype PCs.
- GWAS results include effect sizes, p-values, and -log10(p-values).

---

## 📂 Output Directory Structure

```
asset/
└── gwas_results/
    ├── ALL.chr22.phase3_shapeit2_....vcf.gz
    ├── gwas_results.csv
    ├── manhattan_plot.png
    └── qq_plot.png
```

---

## 📧 Contact
For questions or improvements, open an issue or contact the maintainer.
