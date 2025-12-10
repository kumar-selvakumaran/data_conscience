# TEST CHANGE Data Seance - Ethics Compliance Analysis System

## Overview 

**Data Seance** is an integrated ethics compliance analysis system for data science and AI projects. It combines explainable AI (XAI) methods, LLM-based ethics evaluation, and an interactive IDE-like interface to help teams identify and remediate ethical risks in datasets and machine learning models.

The system analyzes projects against customizable ethical guidelines and provides:
- **Automated ethics audits** using LLM evaluation
- **Explainability artifacts** (ICE plots, LIME explanations)
- **Structured findings** with evidence-based recommendations
- **Interactive UI** for exploring and expanding on issues
- **Contextual guidance** with citations and resources

**Team Members:**
- Purvaja Narayana 
- Kumar Selvakumaran 

---

## Architecture

### Backend (Python/Flask)

Located in [`backend/`](backend)

**Core Modules:**

- `main.py` - Main entry point; orchestrates analysis pipeline
- `api.py` - Flask REST API for frontend communication
- `llm_call.py` - LLM integration (OpenRouter)
- `llm_helpers.py` - XAI methods (ICE plots, LIME) and data descriptions
- `prompts.py` - System prompts for ethics compliance agents
- `run_api.py` - API server entry point

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `analyze_ethics()` | Unified entry point for dataset/model analysis |
| `analyze_ethics_compliance()` | Full compliance analysis with XAI artifacts |
| `elaborate_attribute()` | Deep-dive into a single ethical finding |
| `describe_pandas_dataset()` | Generate LLM-friendly dataset descriptions |
| `describe_sklearn_model()` | Extract and format sklearn model information |
| `ceteris_paribus_bytes_multi()` | Generate ICE (Individual Conditional Expectation) plots |
| `lime_explain_instances()` | Generate LIME explanations for instances |

### Frontend (React/Vite)

Located in [`frontend/`](frontend)

**Key Components:**

- `App.jsx` - Root application component
- `DataSeancePanel.jsx` - Main analysis panel with citations
- `ProblemsPanel.jsx` - Displays issues, handles attribute expansion
- `Sidebar.jsx` - File explorer and upload
- `EditorArea.jsx` - File viewer
- `ActivityBar.jsx` - View selector
- `TopMenuBar.jsx` - App header
- `StatusBar.jsx` - Status indicators

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Node.js 16+**
- **OpenRouter API Key** (for LLM access)

### Backend Setup

```bash
# 1. Navigate to project root
cd Data_Seance

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

**Key Dependencies:**
- `langchain-openai` - LLM integration
- `pandas`, `numpy`, `scikit-learn` - Data science
- `lime` - Local explanations
- `flask`, `flask-cors` - REST API
- `matplotlib` - Visualization

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Start dev server
npm run dev
```

The frontend will run on `http://localhost:5173`

---

## Running the System

### Start Backend API

```bash
# From project root
python backend/run_api.py
```

The API server runs on `http://localhost:5001`

**Available Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze` | POST | Analyze CSV file or code |
| `/api/request-details` | POST | Expand a single issue |
| `/api/health` | GET | Health check |
| `/api/cache/status` | GET | View cached analyses |

### Start Frontend

```bash
# From frontend/
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage

### Basic Workflow

1. **Upload a File** - Use the sidebar to upload CSV or code files
2. **Analyze** - System automatically analyzes for ethical issues
3. **Review Results** - Issues appear in the Problems panel with severity badges
4. **Get Clarity** - Click "Give Clarity" on any issue for detailed expansion
5. **Explore Context** - View citations and ethical guidelines

### Example: Dataset Analysis

```python
from backend.main import analyze_ethics
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Run analysis
result = analyze_ethics(
    data=df,
    project_description="Customer segmentation for marketing",
    prompt_style="detailed"
)

# Access results
print(result['llm_response'])  # Full analysis
for issue in result['issues']:
    print(f"- {issue['issue_name']}: {issue['issue_description']}")
```

### Example: Model Analysis with Explainability

```python
from backend.main import analyze_ethics_compliance
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Prepare data
X_train, X_test = ...
y_train, y_test = ...

# Train model
model = RandomForestRegressor().fit(X_train, y_train)

# Analyze with ICE plots and LIME
result = analyze_ethics_compliance(
    model=model,
    data=X_test,
    data_train=X_train,
    project_description="Housing price prediction",
    include_ice=True,
    include_lime=True,
    metadata={
        'CRIM': 'Crime rate by town',
        'RAD': 'Index of accessibility to radial highways'
    }
)

print(result['llm_response'])
```

---

## Ethical Guidelines

The system evaluates projects against customizable ethical guidelines. Default guidelines cover:

1. **Protected Attributes** - Detection of sensitive/protected characteristics
2. **Data Fairness** - Bias and disparate impact assessment
3. **Privacy & Data Governance** - PII, data minimization, compliance
4. **Transparency** - Documentation and explainability
5. **Human Oversight** - Human review mechanisms
6. **Technical Robustness** - Data quality and preprocessing
7. **Accountability** - Audit trails and versioning
8. **Security** - Adversarial robustness, malicious use prevention
9. **Deployment Practices** - Monitoring, graceful degradation
10. **Research Practices** - Realistic claims, reproducibility

See [`assets/guidelines/guidelines_shorter.txt`](assets/guidelines/guidelines_shorter.txt) for full guidelines.

---

## Analysis Output

### Structured Issues

Each issue follows this format:

```
[NUMBER]. [CATEGORY]
Status: [Violation | Possible Concern | Compliant | Not Assessable]
Description: [One clear sentence explaining the concern]
Evidence:
- [Specific factual observation]
- [Quantitative metric if available]
Recommendation:
- [Actionable remedy with priority]
```

### Expanded Analysis (Attribute Expansion)

When requesting clarity on an issue, the system provides:

- **Expanded Rationale** - Why the issue matters and which guidelines it violates
- **Impact & Stakeholders** - Who is affected and how
- **Root Cause Hypotheses** - Likely technical causes
- **Diagnostics to Run** - Concrete checks to validate concerns
- **Detailed Remediation Plan** - Short-term and long-term fixes

---

## API Reference

### POST `/api/analyze`

Analyze a file for ethical compliance.

**Request:**
```json
{
  "file_name": "data.csv",
  "file_content": "...",
  "project_description": "Brief description of the project"
}
```

**Response:**
```json
{
  "success": true,
  "project_name": "data.csv",
  "cache_key": "hash_123",
  "issues": [
    {
      "issue_name": "1. Protected Attributes",
      "issue_description": "...",
      "issue_severity": "error|warning|info",
      "issue_evidence": "...",
      "possible_remedies": [...]
    }
  ],
  "full_analysis": "Complete LLM response"
}
```

### POST `/api/request-details`

Get detailed explanation for a specific issue.

**Request:**
```json
{
  "issue_name": "1. Protected Attributes",
  "issue_description": "...",
  "issue_evidence": "...",
  "file_name": "data.csv",
  "file_content": "...",
  "cache_key": "hash_123"
}
```

**Response:**
```json
{
  "success": true,
  "additional_details": "Markdown formatted expansion with sections..."
}
```

---

## Testing

### Backend Tests

```bash
# Test dataset-only analysis
python test/run_pii.py

# Test model + dataset analysis with attribute expansion
python test/attribute_expansion.py
```

**Test Data:**
- `test/PIIdata.csv` - Sample PII dataset
- `test/boston_housing_dataset.pkl` - Boston Housing regression dataset

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
OPENROUTER_API_KEY=your_key_here
```

### Backend Configuration

In `backend/main.py`:

```python
# Default guidelines path
DEFAULT_GUIDELINES_PATH = Path(__file__).parent.parent / 'assets' / 'guidelines' / 'guidelines_shorter.txt'

# Prompt style options
prompt_style = "detailed"  # or "concise" or "structured"

# Temperature for LLM (0-2)
temperature = 0.7

# Model selection
model_id = None  # Uses default; set to specific model if needed
```

### Customizing Guidelines

Replace the guidelines file or pass custom guidelines:

```python
with open('your_guidelines.txt', 'r') as f:
    custom_guidelines = f.read()

result = analyze_ethics(
    data=df,
    guidelines_text=custom_guidelines
)
```

---

## Performance

### Typical Analysis Times

| Analysis Type | Time |
|---------------|------|
| Dataset-only (no model) | 5-15 seconds |
| Dataset + Model | 15-30 seconds |
| With ICE plots (5 features) | 20-40 seconds |
| With LIME (3 instances) | 25-45 seconds |
| Full analysis (ICE + LIME) | 40-60 seconds |

### Optimization Tips

- Use `include_ice=False` and `include_lime=False` for faster analysis
- Reduce `num_ice_datapoints` and `num_lime_instances` for speed
- Cache results for repeated analyses on same data

---

## Limitations

- **Model Support** - Currently supports scikit-learn models; TensorFlow/PyTorch coming soon
- **Data Size** - Tested up to ~100K rows; larger datasets may be slow
- **ICE Plots** - Works best with numeric features; categorical features require encoding
- **LIME** - Requires training data; not available for dataset-only analysis
- **Guidelines Scope** - Default guidelines focus on fairness, privacy, and transparency; not comprehensive for all ethical domains

---

## Project Structure

```
Data_Seance/
├── backend/
│   ├── __init__.py           # Package initialization
│   ├── api.py                # Flask REST API
│   ├── main.py               # Main analysis orchestration
│   ├── llm_call.py           # LLM integration
│   ├── llm_helpers.py        # XAI methods & data descriptions
│   ├── prompts.py            # System prompts
│   └── run_api.py            # API server entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Root component
│   │   ├── main.jsx          # Entry point
│   │   └── ide/              # IDE components
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── test/
│   ├── run_pii.py            # Dataset-only analysis test
│   ├── attribute_expansion.py # Model + XAI test
│   ├── PIIdata.csv           # Sample dataset
│   └── boston_housing_dataset.pkl
├── assets/
│   └── guidelines/           # Ethical guidelines & references
├── .env.example
├── requirements.txt
└── README.md
```

