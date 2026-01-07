# custom_llm.py (Modified)
import requests
import json # Import json for better payload handling

class CustomFarmAzureLLM:
    def __init__(self, endpoint, deployment, api_version, subscription_key, proxy=None):
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.subscription_key = subscription_key
        self.proxy = proxy

    # Change invoke to accept a list of messages
    def invoke_chat(self, messages_list):
        url = f"{self.endpoint}/api/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        payload = {
            "messages": messages_list, # Use the list passed in
            "max_tokens": 1024,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "genaiplatform-farm-subscription-key": self.subscription_key
        }

        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None

        # Use json.dumps for robustness
        response = requests.post(url, data=json.dumps(payload), headers=headers, proxies=proxies)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
    
    # Keep the old invoke as a wrapper for simple tests (like in initialize_llm)
    def invoke(self, prompt):
        # Default behavior uses a simple system prompt
        messages_list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self.invoke_chat(messages_list)

