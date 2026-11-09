# Connect an optional LLM

[中文](LLM_SETUP_ZH.md)

Use **Connect model** in the top-right corner. Choose DeepSeek, OpenAI, or an OpenAI-compatible provider, then enter its base URL, an available model ID, and API key. Save the settings and explicitly test the connection; saving alone does not establish connectivity.

Example base URLs are `https://api.deepseek.com` and `https://api.openai.com/v1`. A compatible endpoint must provide the expected Chat Completions API. Use model IDs actually supported by the selected account.

Testing sends one provider request. Enabled chat sends text and necessary task summaries; photos and checkpoint bytes are not uploaded with chat. Keys remain in process memory, not the project database or release package. Re-enter after restart or use the startup variables documented in [llm.example.env](../llm.example.env). Do not put real keys in committed files.

The LLM interprets intent and proposes a supported next action. The application checks the actual project state, arguments, data roles, and confirmation. Training and model decisions remain governed by the same workflow. Without an LLM, deterministic task operation continues.

For a failure, inspect the reported HTTP status, timeout, DNS, proxy, or TLS error, then check the base URL, model name, credentials, and provider permissions. GPU setup and PyTorch installation are not required for LLM API access.
