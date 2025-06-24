#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Make API call
curl -X POST "https://api.openrouter.ai/v1/completions" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "modelname", "prompt": "Hello, world!", "max_tokens": 50}'

