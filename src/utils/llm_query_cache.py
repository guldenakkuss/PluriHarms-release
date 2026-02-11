import os
import json
from openai import OpenAI
from tqdm import tqdm

class LLMQueryCache:
    def __init__(self, cache_dir, model):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.llm_cache = os.path.join(cache_dir, f"{model.replace("-", "_")}_cache.jsonl")
        self.client = OpenAI()
        self.model = model
    
    def _log_query(self, prompt, response):
        entry = {
            "prompt": prompt,
            "response": response,
            "model": self.model
        }
        with open(self.llm_cache, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _get_cached_response(self, prompt):
        """Check if a cached response exists for the given prompt."""
        if not os.path.exists(self.llm_cache):
            return None
        
        try:
            with open(self.llm_cache, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("prompt") == prompt and entry.get("model") == self.model:
                        return entry.get("response")
        except (json.JSONDecodeError, IOError):
            return None
        
        return None
    
    def query(self, prompt, force_rerun=False):
        # Check cache first unless force_rerun is True
        if not force_rerun:
            cached_response = self._get_cached_response(prompt)
            if cached_response:
                return cached_response
        
        with tqdm(total=1, desc=f"Querying {self.model}") as pbar:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1
            )
            pbar.update(1)

        response_text = response.choices[0].message.content
        
        self._log_query(prompt, response_text)
        
        return response_text