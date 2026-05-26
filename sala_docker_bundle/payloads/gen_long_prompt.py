para = (
  "The Roman Empire was a remarkably complex civilization that produced enduring "
  "contributions to law, engineering, architecture, language, and political thought. "
  "Stretching at its height from Britain in the northwest to Mesopotamia in the east, "
  "it integrated vastly different peoples under a single legal and administrative system. "
)
n = 190  # target ~12k tokens
text = (para * n) + "\n\nSummarize in two sentences what the passage above describes."
import json
print(json.dumps({"text": text, "sampling_params": {"max_new_tokens": 64, "temperature": 0.0}}))
