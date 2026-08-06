# Copy this file to config.local.ps1, fill in your own key, then run:
# . .\config.local.ps1
# python -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8010 --reload

$env:PYTHONPATH = "src"
$env:DEEPSEEK_API_KEY = "replace-with-your-own-key"

# Set to "1" only if you intentionally want DeepSeek requests to use your system proxy.
$env:DEEPSEEK_USE_PROXY = "0"

# These defaults normally do not need changing.
$env:CAMPUS_RAG_EMBEDDING_INDEX = "data/embedding_index.json"
$env:CAMPUS_RAG_LEXICAL_INDEX = "data/index.json"
