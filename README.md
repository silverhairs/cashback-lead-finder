# Lead Finder

Gets information about companies then scores each one of them based on size and industry fit.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv sync                                          # Install dependencies
```

## Run

```bash
uv run python finperks_lead_finder.py <input file name>
```

or

```bash
python finperks_lead_finder.py <input file name>
```

**Input**: CSV with `Name,Domain` columns
**Output**: `companies_scored.csv` + `companies_funnel.png`

Options: `-o custom.csv` or `--no-viz`

## How It Works

### Input

A CSV file with the companies that are included in the funnel, structured in two columns: `Name` (The company name) and `Domain` (The company's domain name).

### Data Enrichment

The enrichment is currently done by an API mock that generates random data for each company in the initial csv file. The results is then added to the companies and exported into an output csv file.

### Scoring

Scoring is done on the enriched companies, and included to the output csv file.

**Formula**: (Company Size Score + Industry Score) / 2

**Company Size**:

- 501-1,000 employees = 100 pts (optimal)
- 201-500 = 90 pts
- 1,001-5,000 = 85 pts
- 51-200 = 75 pts
- 5,000+ = 70 pts
- <50 = 60 pts

**Industry** (LinkedIn categories):

- Banking, Investment Banking, Capital Markets = 100 pts
- Credit Intermediation, Savings = 70 pts
- Insurance, Trusts = 40 pts
- Other = 30 pts

**Grades**: A (90-100), B (80-89), C (60-79), D (<60)

### Visualization

With the enriched and scored data, we then use [Seanborn](https://seaborn.pydata.org/) to create a visualization of the funnel and export it as an image.
