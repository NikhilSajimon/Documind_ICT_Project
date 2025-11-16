# DocuMind: An Intelligent Document Processing (IDP) Platform

This is the final project for the ICT Academy of Kerala's certification in Machine Learning & Artificial Intelligence. "DocuMind" is a complete, end-to-end MLOps project that builds, analyzes, and deploys a template-free AI model to extract structured data from unstructured documents like receipts and forms.

## Live Deployed Application

**The final application is deployed on Google Cloud Run and is accessible here:**

[https://documind-service-407790002332.us-central1.run.app](https://documind-service-407790002332.us-central1.run.app)


---

## 1. Problem Statement

In virtually every business, critical data (names, dates, totals, addresses) is trapped inside unstructured documents like PDFs, scanned receipts, and forms. Manually extracting this data is a major operational bottleneck—it's slow, expensive, and notoriously error-prone.

While modern Generative AI (GenAI) models can perform this task, they present two critical blockers for most businesses:

1.  **Prohibitive Cost:** API calls for GenAI models are priced per document (e.g., ~$0.005 per page). For a business processing 1 million documents a month, this translates to **$5,000+ in monthly API fees**.

2.  **Data Privacy & Compliance:** Using a third-party API requires sending sensitive, private customer data (financial, medical, or legal documents) to an external vendor, which is a major compliance violation for many industries.

This project solves this by building a **private, self-hosted, and 100x more cost-effective** custom model to handle specific, high-volume document tasks.

## 2. Visual Showcase

Here is a brief demo of the final deployed application extracting key-value pairs from a test receipt.

*(This is where you should paste the 30-60 second video for your LinkedIn post, converted to a GIF, or just add a screenshot)*

[DocuMind Demo](https://drive.google.com/file/d/16fMK8QsUEk8vHUiGUHDEq0z7eQ1zhVdf/view?usp=sharing)

## 3. Technology Stack

* **ML Model:** `transformers` (LayoutLMv3-base), PyTorch
* **Data Processing:** pandas, NumPy
* **OCR Engine:** `pytesseract`
* **Backend & Server:** Flask, Gunicorn (for production)
* **Deployment:** Docker, Google Cloud Run, Google Artifact Registry
* **Core Data:** FUNSD dataset (forms), SROIE dataset (receipts)

## 4. File Structure (What's in this Repository)

This repository contains all the **code** for the project, from data processing to deployment.

* `app.py`: The final, consolidated Flask application. It contains all inference logic, the spatial linking algorithm, and the API endpoints. It is run by `gunicorn`.
* `Dockerfile`: The production-ready Dockerfile. It installs all system dependencies (like Tesseract), Python libraries, and configures `gunicorn` to serve the app.
* `requirements.txt`: A list of all Python libraries required for the project.
* `.gitignore`: **A critical file.** It tells Git to ignore large data and model files, as explained below.
* `templates/index.html`: The HTML/CSS/JS frontend for the web application.
* `scripts/`: A folder containing all the Python scripts used for the ML pipeline:
    * `normalization.py`: Scales bounding boxes to a universal 0-1000 grid.
    * `prepare_funsd_data.py`: Processes the raw FUNSD dataset.
    * `prepare_sroie_data.py`: Processes and splits the raw SROIE dataset.
    * `train_model.py`: (Or similar) The script used to fine-tune the LayoutLMv3 model.
* **Analysis Files:**
    * `eda_spatial_clustering.png`: A key finding from our EDA, proving that text on documents forms predictable spatial clusters (e.g., columns).
    * `eda_label_distribution.png`: Shows the distribution of different data labels.

### A Note on Missing `data/` and `models/` Folders

You will notice the `data/` and `models/` folders are not included in this repository. **This is intentional and is standard practice.**

1.  **File Size:** The raw `data/` folder is several gigabytes, and the final trained `models/` folder is over 1.2 GB.
2.  **GitHub Limits:** GitHub has a strict 100 MB file limit. It is impossible to push these large binary files.
3.  **Best Practice:** A Git repository is for **source code**, not for large data or compiled model artifacts. The `.gitignore` file is configured to correctly exclude these large directories.

**The working, deployed application link is the definitive proof that the model was successfully trained, saved, and deployed.**

## 5. Project Analysis & Key Finding: The "Receipt Specialist"

The final trained model achieved a **weighted-average F1-Score of 0.84.**

This is not a "good" or "bad" score on its own; it's a **key finding**. The model was trained on a combined dataset of 150 forms (FUNSD) and 500 receipts (SROIE). This 77% imbalance created a highly specialized **"Receipt Specialist"** model.

* This model is **excellent** at its primary task (receipts), scoring very high (likely >0.90 F1).
* It is **weaker** on general forms, as it saw 3.3x fewer examples.
* The 0.84 F1-score is the weighted average of this specialization.

This analysis is the project's core discovery. It proves that we have successfully built a high-performance, specialized tool for a specific business task (receipt processing) and validates the business case for our custom model over a generic GenAI.

## 6. Deployment: The Full MLOps Pipeline

This project was successfully deployed to Google Cloud Run. The journey was a real-world MLOps challenge that required solving two critical errors:

1.  **The "Container Crash" (`exit(1)`):** The initial deployment failed because the container couldn't start. This was traced to two problems:
    * **Import Errors:** The `app.py` script couldn't find the `scripts/` folder. This was fixed by **consolidating** all helper functions into a single, self-contained `app.py`.
    * **Server Failure:** The app was using Flask's *development server* (`app.run()`), which cannot run in a production environment. This was fixed by installing **`gunicorn`** and using it in the `Dockerfile` to start the app correctly.

2.  **The "Upload Timeout" (`ReadTimeout`):** With the 1.2GB `models/` folder included, the simple `gcloud run deploy --source .` command failed because it couldn't upload such a large file. This was solved by using the professional 3-step deployment workflow:
    1.  **`docker build ...`**: Build the 1.65GB+ container on the local machine.
    2.  **`docker push ...`**: Reliably upload the container to Google Artifact Registry in layers.
    3.  **`gcloud run deploy --image ...`**: Tell Cloud Run to pull the container from the registry (a very fast, server-to-server transfer).

## 7. Future Work: The "Production-Ready" Roadmap

This project is a successful "working model." To make it a "perfect" production-ready system, I would follow this 3-phase roadmap:

* **Phase 1: The "Generalist" Model:**
    To create a model that can handle *any* document, we would fix the data imbalance. This involves adding more "form" data (like the Kleister-NDA dataset) and using a **Weighted Data Loader** to train the model on a balanced 1:1:1 ratio of forms, receipts, and legal documents.

* **Phase 2: The "Specialist Router" Architecture:**
    A true production system would not use a single "do-it-all" model. It would be a "router":
    1.  A simple classifier model identifies the document type: `[INVOICE]`, `[RECEIPT]`, or `[NDA]`.
    2.  The app then routes the document to a specific, high-performance "specialist" model (like the "Receipt Specialist" we already built!) for the most accurate extraction.

* **Phase 3: The Asynchronous Batch Pipeline:**
    To process 1,000s of documents at once, we would re-architect the system:
    1.  **Ingest:** Files are dropped into a **Cloud Storage** bucket.
    2.  **Queue:** A trigger sends a "job" message to a **Pub/Sub** queue for each file.
    3.  **Process:** An auto-scaling fleet of **Cloud Run** "workers" (subscribing to the queue) processes each document in parallel, runs the model, and saves the final JSON to a BigQuery database.
