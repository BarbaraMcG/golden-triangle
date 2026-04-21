## Overview

The pipeline follows these main steps:

1. **Data Extraction**: 
    - Download OpenAlex Data Dump
    - Filter for open-access, research articles, proceedings papers or book chapters
    - Categorize and filter to data papers published in 2022, and research papers that at least shared one author and is within 5 years of it being published
    - Filter based on abstract length (shorter than 300 characters or longer than 2,000 characters were excluded)
    - Download full-text content (PDFs/HTML) for those papers
2. **Matching Algorithm**: Run matching algorithms to find related papers. This included:
    - Jaccard
    - TF-IDF
    ...
3. **Experiment Evaluation**: Compare algorithm performance against ground truth matches

## Project Structure
```
NLP Paper/
├── Data Extraction/
│   └── full_text_download.py 
├── Experiments/
│   ├── experiments.py
│   └── tfidf_and_jaccard_experiment.py
├── Matching Algorithms/
│   ├── jaccard_alg.py
│   └── tfidf_alg.py
├── match.xlsx # Ground truth pairs
└── README.md 
```

