import numpy as np

from concordia.contrib import language_models
from concordia.associative_memory import basic_associative_memory
from concordia.prefabs import entity as entity_prefabs
from concordia.utils import helper_functions
from concordia.typing import entity


# --------------------------------------------------
# 1. Set up Llama 3
# --------------------------------------------------

model = language_models.language_model_setup(
    api_type="ollama",
    model_name="llama3:latest",
)


# --------------------------------------------------
# 2. Temporary dummy embedder
# --------------------------------------------------

class DummyEmbedder:
    def __init__(self, dimension=768):
        self._zero_vector = np.zeros(dimension, dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self._zero_vector


embedder = DummyEmbedder()


# --------------------------------------------------
# 3. Create Alice's memory
# --------------------------------------------------

alice_memory = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)

alice_memory.extend([
    "Alice is a 21-year-old computer science student.",
    "Alice is curious and enjoys learning about artificial intelligence.",
    "Alice enjoys dancing and listening to music.",
    "Alice values close friendships.",
    "Alice is ambitious but sometimes overthinks important decisions.",
])


# --------------------------------------------------
# 4. Create Bob's memory
# --------------------------------------------------

bob_memory = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)

bob_memory.extend([
    "Bob is a 22-year-old economics student.",
    "Bob is outgoing and enjoys meeting new people.",
    "Bob enjoys playing basketball.",
    "Bob values his friendships.",
    "Bob is spontaneous and likes trying new things.",
])


# --------------------------------------------------
# 5. Create the agents
# --------------------------------------------------

prefabs = helper_functions.get_package_classes(entity_prefabs)

alice_prefab = prefabs["basic__Entity"]
bob_prefab = prefabs["basic__Entity"]


alice_prefab.params["name"] = "Alice"
alice_prefab.params["goal"] = (
    "Learn about the world, make thoughtful decisions, "
    "and have meaningful conversations."
)

bob_prefab.params["name"] = "Bob"
bob_prefab.params["goal"] = (
    "Have interesting conversations, meet new people, "
    "and enjoy new experiences."
)


alice = alice_prefab.build(
    model=model,
    memory_bank=alice_memory,
)

bob = bob_prefab.build(
    model=model,
    memory_bank=bob_memory,
)


# --------------------------------------------------
# 6. Conversation loop
# --------------------------------------------------

action_spec = entity.ActionSpec(
    call_to_action="What do you do next?",
    output_type=entity.OutputType.FREE,
)

situation = (
    "Alice and Bob are both sitting in a university coffee shop. "
    "They have never met before. Bob notices Alice working on her laptop."
)

print("\n" + "=" * 60)
print("CONVERSATION")
print("=" * 60)

print("\nInitial situation:")
print(situation)


# Give both agents the initial situation
alice.observe(situation)
bob.observe(situation)


# Keep track of the most recent action
last_action = None


for round_number in range(1, 11):

    print(f"\n{'-' * 60}")
    print(f"ROUND {round_number}")
    print(f"{'-' * 60}")

    # -------------------------
    # Alice's turn
    # -------------------------

    if last_action is not None:
        alice.observe(
            "Bob just did the following:\n" + last_action
        )

    alice_action = alice.act(action_spec)

    print("\nAlice:")
    print(alice_action)

    # -------------------------
    # Bob's turn
    # -------------------------

    bob.observe(
        "Alice just did the following:\n" + alice_action
    )

    bob_action = bob.act(action_spec)

    print("\nBob:")
    print(bob_action)

    # Bob's action becomes the context
    # for Alice's next turn.
    last_action = bob_action
