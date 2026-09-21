from concordia.contrib import language_models

model = language_models.language_model_setup(
    api_type="ollama",
    model_name="llama3:latest",
)

response = model.sample_text(
    "In one sentence, explain what an AI agent is."
)

print(response)