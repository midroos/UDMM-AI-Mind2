import numpy as np
import json
from typing import Dict, List, Tuple, Callable, Any
from scipy.stats import entropy
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Import the new dynamic memory module
from .memory_module import MemoryModule

class AttractorDynamics:
    """Dynamic attractor system for UDMM v4"""

    def __init__(self, dimensionality: int = 5, alpha: float = 0.1, beta: float = 0.3):
        self.dimensionality = dimensionality
        self.alpha = alpha
        self.beta = beta
        self.attractor_state = np.random.dirichlet(np.ones(dimensionality))
        self.attractor_labels = ["Ego", "Social", "Symbolic", "Physical", "Cultural"]
        self.history = [self.attractor_state.copy()]
        self.dynamics_matrix = np.random.randn(dimensionality, dimensionality) * 0.1
        np.fill_diagonal(self.dynamics_matrix, 1.0)
        self.coupling_matrix = np.random.randn(dimensionality, dimensionality) * 0.1
        np.fill_diagonal(self.coupling_matrix, 0.0)

    def update(self, external_influence: np.ndarray, internal_tension: float) -> np.ndarray:
        influence_effect = self.beta * external_influence
        dynamic_effect = self.dynamics_matrix @ self.attractor_state
        coupling_effect = self.coupling_matrix @ self.attractor_state
        tension_effect = internal_tension * self.attractor_state
        delta = self.alpha * (influence_effect + dynamic_effect + coupling_effect - tension_effect)
        new_state = self.attractor_state + delta
        new_state = np.clip(new_state, 0.0, 1.0)
        new_state = new_state / np.sum(new_state)
        self.history.append(new_state.copy())
        self.attractor_state = new_state
        return new_state

    def get_attractor_state(self) -> np.ndarray:
        return self.attractor_state

    def plot_attractor_evolution(self, save_path: str = None):
        plt.figure(figsize=(12, 6))
        history_array = np.array(self.history)
        for i in range(self.dimensionality):
            plt.plot(history_array[:, i], label=self.attractor_labels[i])
        plt.title("Attractor State Evolution")
        plt.xlabel("Time Steps")
        plt.ylabel("Attractor Strength")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class IntentManager:
    """Manages hierarchical intentions (structural, self, symbolic)"""

    def __init__(self, dimensionality: int = 5):
        self.dimensionality = dimensionality
        self.intent_hierarchy = {
            "structural": np.random.dirichlet(np.ones(dimensionality)),
            "self": np.random.dirichlet(np.ones(dimensionality)),
            "symbolic": np.random.dirichlet(np.ones(dimensionality))
        }
        self.intent_history = {k: [v.copy()] for k, v in self.intent_hierarchy.items()}

    def update_intent(self, attractor_state: np.ndarray, external_influence: np.ndarray,
                     internal_tension: float, learning_rate: float = 0.1):
        for level, influence_factor in {"structural": 1.0, "self": 1.5, "symbolic": 2.0}.items():
            delta = learning_rate * (
                (attractor_state * influence_factor) +
                (external_influence * 0.5) -
                (internal_tension * self.intent_hierarchy[level])
            )
            new_intent = self.intent_hierarchy[level] + delta
            new_intent = np.clip(new_intent, 0.0, 1.0)
            new_intent = new_intent / np.sum(new_intent)
            self.intent_history[level].append(new_intent.copy())
            self.intent_hierarchy[level] = new_intent

    def get_intent_hierarchy(self) -> Dict[str, np.ndarray]:
        return self.intent_hierarchy

class PromptBuilder:
    """Builds prompts for LLM based on current state. Now memory-agnostic."""

    def __init__(self, attractor_dynamics: AttractorDynamics,
                intent_manager: IntentManager):
        self.attractor_dynamics = attractor_dynamics
        self.intent_manager = intent_manager
        self.context_window = []

    def build_prompt(self, user_input: str, memory_context: List[str] = []) -> str:
        attractor_state = self.attractor_dynamics.get_attractor_state()
        intent_hierarchy = self.intent_manager.get_intent_hierarchy()
        context = {
            "attractor_state": dict(zip(self.attractor_dynamics.attractor_labels, attractor_state)),
            "intent_hierarchy": {k: dict(zip(self.attractor_dynamics.attractor_labels, v))
                                for k, v in intent_hierarchy.items()},
            "user_input": user_input,
            "memory_context": memory_context
        }
        context_str = json.dumps(context, indent=2)
        prompt_template = (
            "You are an AI assistant operating within the UDMM framework.\n"
            "Your responses should reflect your current internal state.\n\n"
            f"Current System State:\n{context_str}\n\n"
            f"User Input:\n{user_input}\n\n"
            f"Memory Context:\n" + "\n".join(memory_context) + "\n\n"
            "Please respond now."
        )
        self.context_window.append({
            "timestamp": len(self.context_window),
            "user_input": user_input,
            "prompt": prompt_template,
            "state": context
        })
        return prompt_template

class UDMMCore:
    """Core of UDMM v4 system with Dynamic Memory Module"""

    def __init__(self, dimensionality: int = 5):
        self.dimensionality = dimensionality
        self.attractor_dynamics = AttractorDynamics(dimensionality=dimensionality)
        self.intent_manager = IntentManager(dimensionality=dimensionality)
        # PromptBuilder is now simpler, without its own memory
        self.prompt_builder = PromptBuilder(self.attractor_dynamics, self.intent_manager)
        # The new dynamic memory module
        self.memory = MemoryModule()

        # System state
        self.time_step = 0
        self.history = []
        # Agent's learnable parameters (placeholder)
        self.theta = np.random.rand(dimensionality)

        # LLM abstraction layer (placeholder)
        self.llm = None

    def process_input(self, user_input: str) -> str:
        print(f"\n--- Time Step {self.time_step} ---")
        # --- 1. SENSE & PREDICT ---
        # Create a placeholder for sensory-driven external influence
        external_influence = np.random.dirichlet(np.ones(self.dimensionality))
        # The "prior" is the agent's attractor state before processing the new input
        p_prior = self.attractor_dynamics.get_attractor_state()

        # --- 2. UPDATE INTERNAL STATE ---
        # A simplified "internal tension" based on input complexity
        internal_tension = min(1.0, len(user_input.split()) / 10.0)

        # Update attractors and intents based on the new influence and tension
        self.attractor_dynamics.update(external_influence, internal_tension)
        self.intent_manager.update_intent(
            self.attractor_dynamics.get_attractor_state(),
            external_influence,
            internal_tension
        )

        # --- 3. DYNAMIC MEMORY STEP ---
        # The "posterior" is the agent's state after updating
        q_post = self.attractor_dynamics.get_attractor_state()
        # The "error" is the difference between the external world and the prior belief
        error = external_influence - p_prior
        # The "global intention" can be drawn from the intent manager (e.g., symbolic level)
        global_intention_vec = self.intent_manager.get_intent_hierarchy()['symbolic']
        global_intention_scalar = np.mean(global_intention_vec) # Simplify to scalar for now

        # Call the new memory module
        self.theta, memory_it = self.memory.step(
            theta=self.theta,
            q_post=q_post,
            p_prior=p_prior,
            error=error,
            global_intention=global_intention_scalar
        )

        # --- 4. ACT ---
        # For now, memory retrieval is not implemented in the new module.
        # We can pass the new schemas if available.
        memory_context = [f"Schema-{i}" for i in range(len(self.memory.schemas))]

        # Build prompt
        prompt = self.prompt_builder.build_prompt(user_input, memory_context)

        # Get response from LLM (placeholder)
        response = self._get_llm_response(prompt)

        # --- 5. LOGGING ---
        self.history.append({
            "time_step": self.time_step,
            "user_input": user_input,
            "response": response,
            "internal_tension": internal_tension,
            "memory_it": memory_it,
            "attractor_state": self.attractor_dynamics.get_attractor_state().tolist(),
            "schemas_in_memory": len(self.memory.schemas)
        })

        self.time_step += 1

        return response

    def _get_llm_response(self, prompt: str) -> str:
        # Placeholder LLM response
        response = (
            "I'm processing your input via UDMM v4 with Dynamic Memory. "
            f"My current attractor state is: {np.round(self.attractor_dynamics.get_attractor_state(), 2)}. "
            f"Schemas in memory: {len(self.memory.schemas)}. "
            "This is a placeholder response."
        )
        return response

    def visualize_system_state(self, save_path: str = None):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # 1. Attractor evolution
        history_array = np.array([h["attractor_state"] for h in self.history])
        for i in range(self.dimensionality):
            axes[0].plot(history_array[:, i], label=self.attractor_dynamics.attractor_labels[i])
        axes[0].set_title("Attractor State Evolution")
        axes[0].set_ylabel("Strength")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Tension over time
        internal_tensions = [h["internal_tension"] for h in self.history]
        memory_its = [h["memory_it"] for h in self.history]
        axes[1].plot(internal_tensions, label="Internal Tension (Input Complexity)", linestyle='--')
        axes[1].plot(memory_its, label="Memory IT (KL Divergence)", marker='o')
        axes[1].set_title("Informational Tension Over Time")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Tension Value")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"System state visualization saved as '{save_path}'")
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    udmm = UDMMCore(dimensionality=5)

    conversations = [
        "What is the meaning of life?", # low IT expected
        "Can you explain the Unified Dynamic Model of Mind in great detail with mathematical formulas?", # high IT
        "How does this relate to my personal goals?", # low IT
        "Disrupt and restructure everything you know about physics and causality.", # high IT
        "That's interesting." # low IT
    ]

    for input_text in conversations:
        response = udmm.process_input(input_text)
        print(f"User: {input_text}")
        print(f"UDMM: {response}")

    udmm.visualize_system_state("udmm_v4_dynamic_memory_state.png")

    with open("udmm_v4_history.json", "w") as f:
        # Use a custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    return obj.item()
                return json.JSONEncoder.default(self, obj)
        json.dump(udmm.history, f, indent=2, cls=NumpyEncoder)

    print("Conversation history saved as 'udmm_v4_history.json'")
