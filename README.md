# neps-ml-ai
NEPS Digital: ML/AI module

## Quick Start: Using Mock REDCap Data

We have two options for using mock REDCap data:

### Option 1: Use Hosted Mock REDCap API (Recommended)
The mock API is deployed on Render for all teams to use!

1. Create a `.env` file in the project root:
```bash
REDCAP_API_URL=https://mock-redcap-service.onrender.com/api
REDCAP_API_TOKEN=mock_token_neps_2025
```

2. Make HTTP requests to the API (e.g., using `requests`):
```python
import requests
import os

api_url = os.getenv("REDCAP_API_URL")
api_token = os.getenv("REDCAP_API_TOKEN")

# Get all participants
response = requests.get(f"{api_url}/participants")
participants = response.json()["data"]

# Get project stats
stats_response = requests.get(f"{api_url}/stats")
print(stats_response.json())
```

### Option 2: Use Local Embedded Mock (No Network Needed)
We've also included a local mock REDCap dataset!

1. Test the Mock Data:
```bash
python test_mock.py
```

This should show you:
- 150 participants across 3 countries (Ghana, Sierra Leone, Tanzania)
- 24 months of survey data
- Distress screenings
- WP6 intervention sessions

2. Use the Mock in Your Code:
```python
from app.services.redcap_mock import RedCapMockClient

client = RedCapMockClient()

# Get all participants
participants = client.get_participants()

# Filter by country
ghana_participants = client.get_participants(country="Ghana")

# Get survey responses
survey_responses = client.get_survey_responses()

# Export all records
all_records = client.export_records()  # Returns list of dicts
# Or export as CSV
csv_data = client.export_records(format="csv")
```

### 3. Data Available
- **150 participants** (50/year across 3 countries)
- **24 months of monthly self-reports** (85% completion rate)
- **Comprehensive wave data** (6, 12, 18, 24 months)
- **Distress screenings** (10% flagged as high-risk)
- **WP6 intervention data** (20 participants, 8 sessions each)

