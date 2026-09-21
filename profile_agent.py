from concordia.contrib import language_models
from concordia.prefabs import entity as entity_prefabs
from concordia.utils import helper_functions
from concordia.typing import prefab as prefab_lib


# Local Ollama model
model = language_models.language_model_setup(
    api_type="ollama",
    model_name="qwen3:8b",
)


# Get Concordia's built-in prefabs
prefabs = helper_functions.get_package_classes(entity_prefabs)


# Alice's stable profile
alice_profile = """
You are Alice.

You are a 21-year-old college student studying computer science.

Personality:
- Introverted
- Practical
- Thoughtful
- Independent

Values:
- Independence
- Efficiency
- Reliability

Preferences:
- You prefer working independently when possible.
- You enjoy solving technical problems.
- You prefer having time to think before making decisions.

When answering questions or making decisions, remain consistent
with this identity and these preferences.
"""


# Create Alice using Concordia's basic Entity prefab
alice_config = prefab_lib.InstanceConfig(
    prefab="basic__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Alice",
        "description": alice_profile,
    },
)


print(alice_config)
