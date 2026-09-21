from concordia.contrib import language_models

# Create the local Ollama language model
model = language_models.language_model_setup(
    api_type="ollama",
    model_name="llama3:latest",
)

# Define the agent
agent_name = "Alice"

agent_background = """
You are Alice, a 21-year-old college student studying computer science.

You enjoy working independently and tend to value efficiency.
You are friendly but somewhat introverted.
"""

# Give the agent an observation
observation = """
You are working on a group project with three other students.
One person wants everyone to work together in the same room.
Another person wants everyone to work independently and combine
their work later.

What would you prefer and why?
"""

# Construct the agent's prompt
prompt = f"""
You are {agent_name}.

{agent_background}

You are currently experiencing the following situation:

{observation}

Respond as Alice.
Give a natural response based on your background and preferences.
Do not describe yourself from an outside perspective.
"""

# Ask Ollama through Concordia
response = model.sample_text(prompt)

print(f"{agent_name}: {response}")