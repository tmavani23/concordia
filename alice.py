import numpy as np

from concordia.contrib import language_models
from concordia.associative_memory import basic_associative_memory
from concordia.prefabs import entity as entity_prefabs
from concordia.utils import helper_functions
from concordia.typing import entity


# --------------------------------------------------
# 1. Ollama / Llama 3
# --------------------------------------------------

model = language_models.language_model_setup(
    api_type="ollama",
    model_name="llama3:latest",
)


# --------------------------------------------------
# 2. Temporary embedder
# --------------------------------------------------

class DummyEmbedder:
    def __init__(self, dimension=768):
        self._zero_vector = np.zeros(dimension, dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self._zero_vector


embedder = DummyEmbedder()


# --------------------------------------------------
# 3. Alice's memory
# --------------------------------------------------

memory_bank = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)

memory_bank.extend([
    "Alice is a 21-year-old computer science student.",
    "Alice is curious and enjoys learning about artificial intelligence.",
    "Alice enjoys dancing and listening to music.",
    "Alice values close friendships.",
    "Alice is ambitious but sometimes overthinks important decisions.",
])


# --------------------------------------------------
# 4. Get the actual basic Entity prefab
# --------------------------------------------------

prefabs = helper_functions.get_package_classes(entity_prefabs)

alice_prefab = prefabs["basic__Entity"]


# --------------------------------------------------
# 5. Configure Alice
# --------------------------------------------------

alice_prefab.params["name"] = "Alice"
alice_prefab.params["goal"] = (
    "Learn about the world, make thoughtful decisions, "
    "and have meaningful conversations."
)


# --------------------------------------------------
# 6. Build Alice
# --------------------------------------------------

alice = alice_prefab.build(
    model=model,
    memory_bank=memory_bank,
)


# --------------------------------------------------
# 7. Give Alice an observation
# --------------------------------------------------

observations = [
    "Alice has finished studying for the day and has some free time.",
    "Alice receives a message from a close friend asking if she wants to hang out.",
    "Alice realizes she has an important assignment due tomorrow.",
    "Alice finishes her assignment and has some time to herself again.",
]


for i, observation in enumerate(observations, start=1):

    print(f"\n{'=' * 50}")
    print(f"ROUND {i}")
    print(f"{'=' * 50}")

    print("\nObservation:")
    print(observation)

    alice.observe(observation)

    action_spec = entity.ActionSpec(
        call_to_action="What do you do next?",
        output_type=entity.OutputType.FREE,
    )

    action = alice.act(action_spec)

    print("\nAlice's action:")
    print(action)

    # Store what happened as a memory.
    memory_bank.add(
        f"Alice observed: {observation} "
        f"Alice decided to: {action}"
    )
