
# TataPlay Retention Copilot

An AI-powered customer retention system that identifies subscribers at risk of churning and generates personalized retention recommendations using Machine Learning, Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG).

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project aims to identify customers who may cancel their subscriptions and recommend suitable retention strategies.

The TataPlay Retention Copilot combines customer data analysis, machine learning, LLM-based classification, and policy-aware retrieval to support customer retention decisions.

## Key Features

- **Customer Data Analysis:** Processes subscriber, recharge, and ticket data.
- **Feature Engineering:** Extracts useful features for customer retention analysis.
- **Churn Risk Analysis:** Identifies subscribers with different levels of retention risk.
- **LLM Classification:** Classifies customer support tickets using weak, zero-shot, and few-shot prompting.
- **Structured Information Extraction:** Uses Pydantic to validate extracted ticket information.
- **RAG Pipeline:** Retrieves relevant company policies to generate grounded answers.
- **Retention Recommendations:** Generates personalized retention briefs and suggested offers.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Large Language Models (LLMs)
- Prompt Engineering
- Pydantic
- ChromaDB
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- SQLite
- Git and GitHub

## Project Structure

```text
tataplay-retention-copilot/
│
├── README.md
├── requirements.txt
├── run_all.py
├── generate_data.py
├── features.py
├── copilot.py
├── llm_client.py
├── mock_llm.py
│
├── part_a.py
├── part_b.py
├── part_c.py
├── part_d.py
│
└── sql/
    └── queries.sql
```

## Model Evaluation Results

### Ticket Classification

| Classification Method | Accuracy |
| --------------------- | -------: |
| Weak Prompting        |  84.25%* |
| Zero-Shot Prompting   |   84.25% |
| Few-Shot Prompting    |   97.26% |

*Weak prompting accuracy is based on extracting the predicted category from the model's explanatory responses, rather than exact label matching.*

### RAG Evaluation

- In-domain retrieval Hit@3: 100% (12/12)
- Overall retrieval Hit@3: 75% (12/16)
- Unsupported-question abstention: 100% (4/4)

These results are from the project's evaluation dataset and are not production performance guarantees.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chgokul/tataplay-retention-copilot.git
cd tataplay-retention-copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure environment variables

If your LLM integration requires an API key, configure it using a `.env` file or the environment variable expected by your code. Never commit API keys to GitHub.

### 6. Run the project

```bash
python run_all.py
```

## Evaluation and Outputs

The project generates outputs such as:

- Retention briefs
- Ticket classification results
- Structured ticket extractions
- RAG evaluation results

Generated files and private customer data should remain local and should not be uploaded to a public repository.

## Future Improvements

- Build an interactive Streamlit dashboard.
- Deploy the retention copilot as a web application.
- Improve churn prediction using additional customer behaviour features.
- Add monitoring for model accuracy and retrieval quality.
- Integrate a human-support escalation workflow.

## Author

**Gokul Krishna Chowdavarapu**

GitHub: [@chgokul](https://github.com/chgokul)

---

If you find this project interesting, feel free to explore the repository and its implementation.
