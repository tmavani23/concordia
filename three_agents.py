import numpy as np

from concordia.contrib import language_models
from concordia.associative_memory import basic_associative_memory
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions


# ============================================================
# 1. LANGUAGE MODEL
# ============================================================

model = language_models.language_model_setup(
    api_type="ollama",
    model_name="llama3:latest",
)


# ============================================================
# 2. DUMMY EMBEDDER
# ============================================================

class DummyEmbedder:
    def __init__(self, dimension=768):
        self._zero_vector = np.zeros(
            dimension,
            dtype=np.float32,
        )

    def __call__(self, text: str) -> np.ndarray:
        return self._zero_vector


embedder = DummyEmbedder()


# ============================================================
# 3. GET PREFABS
# ============================================================

entity_prefabs_dict = helper_functions.get_package_classes(
    entity_prefabs
)

game_master_prefabs_dict = helper_functions.get_package_classes(
    game_master_prefabs
)

prefabs = {
    **entity_prefabs_dict,
    **game_master_prefabs_dict,
}


# ============================================================
# 4. DEFINE ALICE
# ============================================================

alice = prefab_lib.InstanceConfig(
    prefab="basic__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Alice",
        "goal": (
            "Learn about other people, have meaningful "
            "conversations, and pursue her interests."
        ),
    },
)


# ============================================================
# 5. DEFINE BOB
# ============================================================

bob = prefab_lib.InstanceConfig(
    prefab="basic__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Bob",
        "goal": (
            "Meet interesting people, have fun conversations, "
            "and try new experiences."
        ),
    },
)


# ============================================================
# 6. DEFINE CHARLIE
# ============================================================

charlie = prefab_lib.InstanceConfig(
    prefab="basic__Entity",
    role=prefab_lib.Role.ENTITY,
    params={
        "name": "Charlie",
        "goal": (
            "Observe what is happening around them, "
            "make new connections, and have interesting conversations."
        ),
    },
)


# ============================================================
# 7. DEFINE GAME MASTER
# ============================================================

game_master = prefab_lib.InstanceConfig(
    prefab="generic__GameMaster",
    role=prefab_lib.Role.GAME_MASTER,
    params={
        "name": "Coffee Shop Game Master",
        "acting_order": "fixed",
    },
)


# ============================================================
# 8. DEFINE SIMULATION
# ============================================================

config = prefab_lib.Config(
    default_premise=(
        "Alice, Bob, and Charlie are all sitting in a busy "
        "university coffee shop on a Friday afternoon. "
        "Alice is working on her laptop. "
        "Bob is sitting nearby drinking coffee. "
        "Charlie has just entered the coffee shop and is "
        "looking for a place to sit."
    ),

    # Keep this small while debugging.
    default_max_steps=12,

    prefabs=prefabs,

    instances=[
        alice,
        bob,
        charlie,
        game_master,
    ],
)


# ============================================================
# 9. CREATE SIMULATION
# ============================================================

sim = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
)


# ============================================================
# 10. RUN
# ============================================================

print("\n" + "=" * 70)
print("STARTING THREE-AGENT SIMULATION")
print("=" * 70)

results = sim.play()

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

print("\nResults:")
print(results)
