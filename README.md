# Dynamically Benchmarking Long Context Summaries

Data and Code for the paper "Beyond Fact Retrieval: Dynamically Benchmarking Context Utilization in Long Context Summaries". Accepted for findings at [EACL 2027](https://2027.eacl.org/).

_Large Language Models (LLMs) with extensive context windows offer immense potential for automated summarization of long texts. 
However, their reliability is hindered by the "Lost in the Middle" phenomenon, where information located in the center of the input is often omitted or inaccurately represented. 
While this positional bias is well-documented for shorter texts and retrieval tasks, its impact on summaries of long inputs remains under-explored. 
To address this, we introduce a reference-free evaluation framework designed to assess context utilization in generic and query-focused multi-document summaries across dynamic input lengths.
Applying our framework to inputs of 100,000 tokens in an empirical study, we demonstrate that small state-of-the-art open-weight LLMs continue to exhibit clear positional biases compared to proprietary models, which can be mitigated by employing a hierarchical merge strategy._

## Getting Started
Follow the steps below to set up your environment and reproduce the evaluation results.

### Prerequisites
* **Python**: Version 3.10
* **Package Manager**: UV

Install uv following the official [installation guide](https://docs.astral.sh/uv/getting-started/installation/).
After installing uv, install the python dependencies using `uv sync`. 
If the required python version is not yet installed, it can be installed using `uv python install 3.10`.

### Executing scripts
Any scripts must be executed from the root directory of this repository so that any file paths to the data can be resolved correctly. 
Further, always execute scripts as modules (`python -m src.<script name>`) so that dependencies from other scripts can be imported.

### Environment variables
Access credentials and connections are configured via environment variables. 
The variables used in the context of this study are listed below.

```
# OpenAI Connection 
AZURE_OPENAI_ENDPOINT="<...>"
AZURE_OPENAI_API_KEY="<...>"
OPENAI_API_VERSION="<...>"

# Wikipedia Scraper
WIKI_USERAGENT="ProjectName (email@example.com)"

# Local Model Serving
LLAMASWAP_HOST="http://127.0.0.1:9291/v1"
OLLAMA_HOST="http://127.0.0.1:11434"
```

To load environment variables from a file, when executing a script, use: `--env-file`.
For example, the script for downloading the Wikipedia articles is started as follows:

```bash
uv run --env-file .env python -m src.experiments.corpus.extract_data
```

## Codebase Structure
The code base is structured into source code, prompts, data and results. 
In the following we provide a brief overview over the general structure. 
Notably, some parts of the code were adapted from [apfelphine/decoupling-summary-source-alignments](https://github.com/apfelphine/decoupling-summary-source-alignments).
These are marked accordingly.

* `src`: Contains the source code.
  * `core`: Contains core reusable components
    * `document_alignment`, `document_splitting`, `embedding` & `llm`: Components adapted from an [apfelphine/decoupling-summary-source-alignments](https://github.com/apfelphine/decoupling-summary-source-alignments).
    * ...
  * `experiments`: Contains the code to generate the corpus as well as generate, align and evaluate the summaries.
  * `models`: Contains data models
    * `document_splitting`, `llms`: Components adapted from an [apfelphine/decoupling-summary-source-alignments](https://github.com/apfelphine/decoupling-summary-source-alignments).
    * ...
  * `utils`: Contains utility functions
    * `alignment`, `normalization`, `prompts`, `vis`: Components adapted from an [apfelphine/decoupling-summary-source-alignments](https://github.com/apfelphine/decoupling-summary-source-alignments).
    * ... 
* `prompts`: Contains the prompts used for LLM requests
  * `document_alignment`: Prompts adapted from [apfelphine/decoupling-summary-source-alignments](https://github.com/apfelphine/decoupling-summary-source-alignments).
  * `summarization`: Prompts used to generate summaries with different strategies
* `data`: Contains the Wikipedia-articles used as a basis for the generated summaries, grouped by general topic / domain
  * `<Topic A>`
  * `<Topic B>`
  * ...
* `results`: Contains the results of the evaluation 
  * `corpus.json`: Document to Bin mapping
  * `generic_summaries.json`: Input mapping (bin idx and document order) for generic summaries
  * `query_based_summaries.json`: Input mapping (bin idx, document order & target document) for query-focused summaries
  * `summaries`: Contains the generated summaries, document and summary chunks as well as generated alignment matrices

## Running the analysis
In the following, the steps required to run the analysis are detailed.
All data and summaries as reported in the paper are included as part of the paper. 
These steps explain, how they were created.

| Step                                      | Skript                                          | Result                                                                       |
|-------------------------------------------|-------------------------------------------------|------------------------------------------------------------------------------|
| Scrape wikipedia articles                 | `src/experiments/corpus/extract_data.py`        | `data` directory                                                             |
| Group wikipedia articles into bins        | `src/experiments/corpus/transform_bins.py`      | `results/corpus.json`                                                        |
| Define summary inputs per bin             | `src/experiments/corpus/transform_summaries.py` | `results/generic_summaries.json`, </br> `results/query_based_summaries.json` |
| Generate summaries and alignment matrices | `src/experiments/execute.py`                    | `results/summaries` directory                                                |

The different evaluations for the summaries can be found under `src/experiments/<summary-type>/evaluation` respectively.
Generated graphs will be placed under `results/evaluation`.