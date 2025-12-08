# Assignment Evaluator System

A modular, production-ready system for automated evaluation of student assignments using LLMs and course context from MongoDB. The system pulls course content ("course_chunks") and assignment submissions, builds context-rich prompts, and evaluates assignments using a configurable LLM backend (Ollama, Bedrock, or hybrid).

## Features
- Pulls course content and assignment data from MongoDB
- Builds context-aware prompts for LLM-based evaluation
- Supports re-evaluation and manual scan triggers
- Persists evaluation results, feedback, and debug info
- Configurable via YAML (Mongo, LLM, logging, etc.)
- Background scanner/worker for async processing
- FastAPI-based HTTP API for orchestration and results

## Architecture Overview
- **FastAPI app** (`app.py`): Exposes HTTP endpoints for health, evaluation, re-evaluation, results, and admin scan.
- **Worker** (`module/worker.py`): Core logic for prompt building, LLM calls, scoring, and result normalization.
- **Scanner** (`module/scanner_db_runner.py`): Background thread/process that claims and processes queued assignments.
- **LLM Connector** (`llm_model/llm_connector.py`): Unified interface for calling Ollama, Bedrock, or hybrid LLMs.
- **Prompt Builders** (`llm_model/system_instructions.py`): Assembles system prompts from course context and rubric.
- **MongoDB**: Stores assignments, course content, rubrics, and evaluation results.
- **Config**: All settings in `settings/config.yml`.

## API Endpoints

### Health Check
- **GET /health**
  - Returns status of the API and MongoDB connection.

### Assignment Evaluation
- **POST /assignments/{assignment_id}/evaluate?force=false**
  - Queues an assignment for evaluation. If `force=true`, triggers evaluation even if already evaluated.
  - Request body: None
  - Response: `{ "status": "queued_for_evaluation", "job_id": ... }`

- **POST /assignments/{assignment_id}/re-evaluate**
  - Forces re-evaluation of an assignment, regardless of previous status.
  - Request body: None
  - Response: `{ "status": "queued_for_re_evaluation", "job_id": ... }`

### Assignment Results
- **GET /assignments/{assignment_id}/results**
  - Fetches the evaluation results and raw document for an assignment.
  - Response includes: `scores`, `overall`, `feedback`, `violations`, and the full (or truncated) raw document.

### Admin/Debug
- **POST /admin/scan-evaluate**
  - Triggers a manual scan-and-process pass (useful for debugging or batch processing).
  - Response: `{ "processed": <number_of_assignments_processed> }`

## Evaluation Workflow
1. **Queue Assignment**: Use `/assignments/{assignment_id}/evaluate` or `/re-evaluate` to queue an assignment.
2. **Background Scanner**: The scanner thread/process picks up queued assignments and invokes the worker.
3. **Worker**: Builds a system prompt using course context, rubric, and submission. Calls the LLM and normalizes results.
4. **Results**: Evaluation results are stored in MongoDB and can be fetched via `/assignments/{assignment_id}/results`.
5. **Debugging**: Debug info and prompt previews are persisted in the document's `debug.prompts` array for traceability.

## Configuration
- All settings are in `settings/config.yml` (MongoDB, LLM provider, logging, etc.).
- MongoDB connection is managed in `utils/mongo_client.py`.
- LLM provider and prompt size can be tuned via config.

## Scripts
- `scripts/print_prompt.py <doc_id_or_assignment_id>`: Print the system prompt and chunk stats for a given assignment (for debugging prompt construction).

## Requirements
- Python 3.9+
- MongoDB (see config for connection details)
- Ollama or AWS Bedrock for LLM backend (configurable)
- See `requirements.txt` (not shown here) for dependencies

## Notes
- The system is designed for modularity and traceability. All major steps log to file and/or MongoDB.
- For advanced debugging, inspect the `debug.prompts` array in each assignment document.
- Extend or adapt the worker and prompt builders for new evaluation logic or rubric formats.

