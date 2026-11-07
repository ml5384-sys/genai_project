#!/usr/bin/env bash
# Download the two public Kaggle datasets into data/raw (no Kaggle login needed for public datasets;
# if your network requires it, use `kaggle datasets download` with an API token instead).
set -euo pipefail
mkdir -p data/raw/resume data/raw/linkedin
curl -L -o data/raw/resume.zip   https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset
curl -L -o data/raw/linkedin.zip https://www.kaggle.com/api/v1/datasets/download/arshkon/linkedin-job-postings
unzip -q -o data/raw/resume.zip   -d data/raw/resume
unzip -q -o data/raw/linkedin.zip -d data/raw/linkedin
rm data/raw/*.zip
echo "done: $(ls data/raw/resume/Resume) $(ls data/raw/linkedin | head -3)"
