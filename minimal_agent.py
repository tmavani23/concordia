from concordia.contrib import language_models
from concordia.prefabs import entity as entity_prefabs
from concordia.utils import helper_functions
from concordia.typing import prefab as prefab_lib


# Load the local Ollama model
model = language_models.language_model_setup(
    api_type="ollama",
    model_name="qwen3:8b",
)


# Load Concordia's built-in entity prefabs
prefabs = helper_functions.get_package_classes(entity_prefabs)


# Create an Alice instance
alice_config = prefab_lib.InstanceConfig(
    prefab="minimal__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Alice",
    },
)

print("Available entity prefabs:")
print(list(prefabs.keys()))

print("\nAlice configuration created:")
print(alice_config)
