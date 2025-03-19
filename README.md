Setting up environment
pip install -r requirements 

DNA sequence analysis

Running the API
cd ./scripts
uvicorn dna_sequence:app --host 0.0.0.0 --port 8000 --reload

Open new terminal and check if the API is running
curl -X GET "http://127.0.0.1:8000/"

Expected response:
{"message":"Welcome to the Bioinformatics REST API"}

Calculate GC Content
curl -X POST "http://127.0.0.1:8000/gc-content/" -H "Content-Type: application/json" -d "{\"sequence\": \"ATGCGC\"}"

Expected response:
{"sequence":"ATGCGC","gc_content":66.67}

Get reverse complement
curl -X POST "http://127.0.0.1:8000/reverse-complement/" -H "Content-Type: application/json" -d "{\"sequence\": \"ATGCGC\"}"

Expected Response:
{"sequence":"ATGCGC","reverse_complement":"GCGCAT"}

Transcribe DNA to RNA
curl -X POST "http://127.0.0.1:8000/transcribe/" -H "Content-Type: application/json" -d "{\"sequence\": \"ATGCGC\"}"

Expected Response:
{"sequence":"ATGCGC","transcription":"AUGCGC"}


Nuclei segmentation 

Running the API
cd ./scripts
uvicorn nuclei_api:app --host 0.0.0.0 --port 8000 --reload

Upload an Image for Analysis
curl -X POST "http://127.0.0.1:8000/analyze-image/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@./scripts/microscopic_sample.jpg"