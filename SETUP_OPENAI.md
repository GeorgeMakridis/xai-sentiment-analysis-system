# Setting Up OpenAI API Key

The system is currently using **fallback responses** because the OpenAI API key is not configured.

## Quick Fix

### Option 1: Set Environment Variable (Recommended)

1. **Export the API key in your terminal:**
   ```bash
   export OPENAI_API_KEY="sk-your-actual-api-key-here"
   ```

2. **Restart the services:**
   ```bash
   docker-compose restart ai_outputs
   ```

### Option 2: Create .env File

1. **Create a `.env` file in the project root:**
   ```bash
   echo "OPENAI_API_KEY=sk-your-actual-api-key-here" > .env
   ```

2. **Restart the services:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Option 3: Set in docker-compose.yml (Not Recommended)

You can hardcode it in `docker-compose.yml`, but this is less secure.

## Verify It's Working

After setting the API key, check the logs:
```bash
docker-compose logs ai_outputs | grep -i "openai"
```

You should see:
- `"OpenAI API key configured successfully."`
- `"DEBUG: Using OpenAI: True"` (instead of False)

## Get Your API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. Use it in one of the methods above

## Current Status

The system is working but using **fallback responses** which are:
- ✅ Functional
- ✅ Based on your data
- ❌ Less detailed than OpenAI responses
- ❌ Not as contextually aware

With OpenAI API key configured, you'll get:
- ✅ More detailed and contextual responses
- ✅ Better understanding of your questions
- ✅ More natural language interactions
- ✅ Enhanced RAG capabilities
