import dataclasses

from concordia.agents import entity_agent
from concordia.components.agent import (
    memory,
    observation,
    instructions,
    concat_act_component,
)
from concordia.typing import prefab as prefab_lib
from concordia.contrib import language_models

from concordia.associative_memory import basic_associative_memory


# Create a memory bank for Alice
memory_bank = basic_associative_memory.AssociativeMemoryBank(
    embedder=None,
)


# Create the local Ollama model
model = language_models.language_model_setup(
    api_type="ollama",
    model_name="qwen3:8b",
)


@dataclasses.dataclass
class AliceAgent(prefab_lib.Prefab):
    """A simple Concordia agent representing Alice."""

    def build(self, model, memory_bank):
        agent_name = "Alice"

        # 1. Memory
        mem = memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # 2. Recent observations
        obs = observation.LastNObservations(
            history_length=10
        )

        # 3. Instructions / identity
        inst = instructions.Instructions(
            agent_name=agent_name
        )

        # Put the components together
        components = {
            mem.name: mem,
            obs.name: obs,
            inst.name: inst,
        }

        # 4. Action component
        act_component = concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        # Create the actual EntityAgent
        return entity_agent.EntityAgent(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components,
        )
