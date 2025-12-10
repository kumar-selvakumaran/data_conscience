# Data Conscience

1. v1 rollout
    - Technical deliverables: ReACT agent using explainability tools, and guidelines.
    - UI : executed jupyter notebooks.
    - Milestones:
        - Langgraph UI tracing + maximum detail tracing. (tokens, cost, rates-limits, execution times, time to first token, time to each token, reasoning, thinking, and more)
        - Implement `HOST` and `CLIENT` using MCP.

## END GOAL

### Elements

1. **HOST** : This the Agent that is deployed in the cloud.
2. **CLIENT** : A component that the user should download and configure, that requires permissibilty settings. It controls the information flow to and from the host.

### Situation

- There is only a single host, that can be used on demand by any number of clients.
- The user will have a data science project they want to ensure follows their company's ethical guidelines. They decide to use the *Data Conscience* agent.
- The user's project's code is "confidential". Certain aspects of the client's code and data. should not exit the local enivronment (should not reach the host).

### Workflow

1. Based on permissibilty preferences, the CLIENT structures the available context and sends it to HOST.
2. The HOST runs explainability analsis, and parses the context. In case the HOST requires more information, It generates EDA code that it requests to be run on the User's project.
3. The client decides on whether or not to run the Host's EDA code on the user's machine, and how much of the output it should return.
4. This EDA feedback loop goes on till the HOST has all the information it needs. The HOST then renders the final report containing any violations, and it's reasoning behind the claims, that is accessible to the user.

## Project structure Guide

```text

your-agent-project/
├── src/
│   ├── app/
│   │   ├── api/              # FastAPI routers, HTTP handlers
│   │   │   ├── v1/
│   │   │   │   ├── agents.py
│   │   │   │   ├── tasks.py
│   │   │   │   └── health.py
│   │   ├── core/             # App-level concerns
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   ├── security.py   # auth, rate limiting, permissions
│   │   │   └── dependencies.py
│   │   └── main.py           # FastAPI app factory / entrypoint
│   │
│   ├── agents/
│   │   ├── base.py           # BaseAgent, typing, shared utilities
│   │   ├── registry.py       # register_agent(), get_agent()
│   │   ├── single/
│   │   │   ├── job_matcher.py
│   │   │   └── researcher.py
│   │   └── multi/
│   │       ├── crews.py      # crewAI crews or LangGraph node compositions
│   │       └── orchestrators.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_scraping.py
│   │   ├── llm_tools.py      # embeddings, summarization, classification
│   │   ├── db_tools.py
│   │   └── external_apis/
│   │       ├── github.py
│   │       └── job_boards.py
│   │
│   ├── workflows/            # Graphs / flows, separate from agent definitions
│   │   ├── langgraph/
│   │   │   ├── graphs.py     # Graph definitions (nodes, edges)
│   │   │   └── state.py      # State schema
│   │   └── crewai/
│   │       └── crews.py      # Optional: crew definitions
│   │
│   ├── domain/               # Pure business logic
│   │   ├── models.py         # Pydantic models
│   │   ├── services.py       # Core business rules
│   │   └── repositories.py   # DB access or gateways
│   │
│   ├── infra/                # Infrastructure adapters
│   │   ├── llm/
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── gemini_client.py
│   │   ├── vectorstore/
│   │   │   ├── pgvector.py
│   │   │   └── chroma.py
│   │   ├── persistence/
│   │   │   ├── sessions.py   # agent run state
│   │   │   └── events.py     # event sourcing / traces
│   │   └── messaging/
│   │       └── queue.py      # e.g., Redis, Kafka for background jobs
│   │
│   ├── ui/
│   │   ├── streamlit_app.py  # or Gradio, or simple Next.js client
│   │   └── components/
│   │       └── chat.py
│   │
│   └── evals/
│       ├── scenarios/
│       │   ├── job_matching.jsonl
│       │   └── bugfixing.jsonl
│       ├── metrics.py
│       └── run_evals.py
│
├── tests/
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_tools.py
│   │   └── test_domain.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_workflows.py
│   │   └── test_llm_integration.py
│   └── e2e/
│       └── test_full_scenarios.py
│
├── configs/
│   ├── default.yaml
│   ├── dev.yaml
│   └── prod.yaml
│
├── scripts/
│   ├── seed_data.py
│   ├── migrate.py
│   └── run_local_agent.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       ├── ci.yml            # tests + lint + typecheck
│       └── deploy.yml        # optional: build & deploy
│
├── pyproject.toml            # dependencies + tooling
├── README.md
├── CONTRIBUTING.md
└── LICENSE


```