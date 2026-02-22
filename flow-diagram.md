# ReAct LangChain Program Flow

## Block Diagram

```mermaid
flowchart TB
    subgraph init["Initialization"]
        A[main()] --> B[Define tool_list: get_text_length]
        B --> C[Create ReAct PromptTemplate]
        C --> D[Create ChatOllama LLM with AgentCallbackhandler]
        D --> E[Build LCEL Agent Chain]
    end

    subgraph chain["LCEL Agent Chain"]
        direction LR
        F["input"] --> G["format_log_to_str\n(scratchpad)"]
        G --> H["PromptTemplate"]
        H --> I["ChatOllama"]
        I --> J["ReActSingleInputOutputParser"]
    end

    subgraph loop["Main Agent Loop"]
        K[Start Loop] --> L[Invoke Agent: question + scratchpad]
        L --> M{AgentCallbackhandler}
        M --> |on_llm_start| N[Print prompt to LLM]
        M --> |on_llm_end| O[Print LLM response]
        N --> P[Parse Output]
        O --> P
        P --> Q{Result Type?}
        Q -->|AgentAction| R[find_tool_by_name]
        R --> S[Execute Tool: get_text_length]
        S --> T[Append to intermediate_steps]
        T --> L
        Q -->|AgentFinish| U[Print return_values]
        U --> V[End]
    end

    E --> K
```

## Detailed Flow

### 1. Setup Phase (`main()`)
| Component | Purpose |
|-----------|---------|
| `get_text_length` | Tool that returns character count of input string |
| `tool_list` | List containing the single tool |
| `myPrompt` | ReAct-style template with Thought/Action/Observation format |
| `myLlm` | ChatOllama (gemma3) with `AgentCallbackhandler` for logging |
| `myAgent` | LCEL pipeline: input → format scratchpad → prompt → LLM → parser |

### 2. Agent Loop (ReAct Cycle)
```text
┌─────────────────────────────────────────────────────────────────┐
│  Invoke Agent                                                    │
│  Input: "What is the length of the string DOG?"                  │
│  Scratchpad: intermediate_steps (initially [])                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgentCallbackhandler (callback.py)                              │
│  • on_llm_start → prints prompt being sent                       │
│  • on_llm_end   → prints LLM response text                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  ReActSingleInputOutputParser                                    │
│  Parses LLM output → AgentAction OR AgentFinish                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
┌──────────────────────┐      ┌──────────────────────┐
│  AgentAction         │      │  AgentFinish          │
│  (tool + tool_input)  │      │  (return_values)      │
└──────────┬───────────┘      └──────────┬────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Run tool, append     │      │  Print answer,       │
│  (action, obs) to     │      │  EXIT                │
│  intermediate_steps   │      └──────────────────────┘
│  LOOP back ↑          │
└──────────────────────┘
```

### 3. Data Flow Summary

```text
User Question ──► PromptTemplate ──► ChatOllama ──► Parser
      │                │                  │            │
      │                │                  │            ├── AgentAction ──► Tool ──► Observation ──► (back to scratchpad)
      │                │                  │            │
      └── scratchpad ──┘                  │            └── AgentFinish ──► Done
                                          │
                              AgentCallbackhandler (logs I/O)
```
