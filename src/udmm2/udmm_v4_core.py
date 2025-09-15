import numpy as np
import json
from typing import Dict, List, Tuple, Callable, Any
from scipy.stats import entropy
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from scipy.spatial import distance

class AttractorDynamics:
    """Dynamic attractor system for UDMM v4"""

    def __init__(self, dimensionality: int = 5, alpha: float = 0.1, beta: float = 0.3):
        """
        Initialize attractor dynamics system

        Parameters:
        - dimensionality: number of attractor dimensions (ego, social, symbolic, physical, cultural)
        - alpha: learning rate for attractor evolution
        - beta: influence coefficient for external inputs
        """
        self.dimensionality = dimensionality
        self.alpha = alpha
        self.beta = beta

        # Initial attractor state (normalized)
        self.attractor_state = np.random.dirichlet(np.ones(dimensionality))

        # Attractor labels for interpretation
        self.attractor_labels = ["Ego", "Social", "Symbolic", "Physical", "Cultural"]

        # Attractor history for visualization
        self.history = [self.attractor_state.copy()]

        # Attractor dynamics matrix (how attractors influence each other)
        self.dynamics_matrix = np.random.randn(dimensionality, dimensionality) * 0.1
        np.fill_diagonal(self.dynamics_matrix, 1.0)  # Self-influence

        # Attractor coupling matrix (how attractors interact)
        self.coupling_matrix = np.random.randn(dimensionality, dimensionality) * 0.1
        np.fill_diagonal(self.coupling_matrix, 0.0)  # No self-coupling

    def update(self, external_influence: np.ndarray, internal_tension: float) -> np.ndarray:
        """
        Update attractor state based on external influence and internal tension

        Parameters:
        - external_influence: vector of external influences (normalized)
        - internal_tension: scalar representing informational tension

        Returns:
        - Updated attractor state
        """
        # 1. Apply external influence (scaled by beta)
        influence_effect = self.beta * external_influence

        # 2. Apply internal dynamics (attractor evolution)
        dynamic_effect = self.dynamics_matrix @ self.attractor_state

        # 3. Apply coupling between attractors
        coupling_effect = self.coupling_matrix @ self.attractor_state

        # 4. Apply tension-driven adjustment (higher tension = more change)
        tension_effect = internal_tension * self.attractor_state

        # 5. Combine all effects with learning rate
        delta = self.alpha * (influence_effect + dynamic_effect + coupling_effect - tension_effect)

        # 6. Update attractor state
        new_state = self.attractor_state + delta
        new_state = np.clip(new_state, 0.0, 1.0)  # Ensure values between 0-1
        new_state = new_state / np.sum(new_state)  # Normalize

        # 7. Store in history
        self.history.append(new_state.copy())
        self.attractor_state = new_state

        return new_state

    def get_attractor_state(self) -> np.ndarray:
        """Get current attractor state"""
        return self.attractor_state

    def get_attractor_influence(self, context_vector: np.ndarray) -> float:
        """Calculate how much attractors influence the current context"""
        # Simple dot product between attractor state and context vector
        return np.dot(self.attractor_state, context_vector)

    def plot_attractor_evolution(self, save_path: str = None):
        """Plot attractor evolution over time"""
        plt.figure(figsize=(12, 6))

        # Convert history to array for plotting
        history_array = np.array(self.history)

        # Plot each attractor dimension
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

    def calculate_informational_tension(self, new_context: np.ndarray) -> float:
        """
        Calculate informational tension between current state and new context

        Parameters:
        - new_context: context vector to compare against

        Returns:
        - Informational tension value
        """
        # Simple KL divergence between attractor state and context
        # Add small epsilon to avoid division by zero
        p = self.attractor_state + 1e-10
        q = new_context + 1e-10
        return entropy(p, q)

class IntentManager:
    """Manages hierarchical intentions (structural, self, symbolic)"""

    def __init__(self, dimensionality: int = 5):
        self.dimensionality = dimensionality
        self.intent_hierarchy = {
            "structural": np.random.dirichlet(np.ones(dimensionality)),
            "self": np.random.dirichlet(np.ones(dimensionality)),
            "symbolic": np.random.dirichlet(np.ones(dimensionality))
        }
        self.intent_history = {
            "structural": [self.intent_hierarchy["structural"].copy()],
            "self": [self.intent_hierarchy["self"].copy()],
            "symbolic": [self.intent_hierarchy["symbolic"].copy()]
        }

    def update_intent(self, attractor_state: np.ndarray, external_influence: np.ndarray,
                     internal_tension: float, learning_rate: float = 0.1):
        """
        Update intentions based on attractor state and external influences

        Parameters:
        - attractor_state: current attractor state
        - external_influence: external influence vector
        - internal_tension: current informational tension
        - learning_rate: how quickly intentions change
        """
        # Update each level of intention
        for level in ["structural", "self", "symbolic"]:
            # Higher levels are more influenced by attractors
            influence_factor = 1.0 if level == "structural" else (1.5 if level == "self" else 2.0)

            # Calculate delta for this level
            delta = learning_rate * (
                (attractor_state * influence_factor) +
                (external_influence * 0.5) -
                (internal_tension * self.intent_hierarchy[level])
            )

            # Update and normalize
            new_intent = self.intent_hierarchy[level] + delta
            new_intent = np.clip(new_intent, 0.0, 1.0)
            new_intent = new_intent / np.sum(new_intent)

            # Store in history
            self.intent_history[level].append(new_intent.copy())
            self.intent_hierarchy[level] = new_intent

    def get_intent_hierarchy(self) -> Dict[str, np.ndarray]:
        """Get current intent hierarchy"""
        return self.intent_hierarchy

    def plot_intent_evolution(self, save_path: str = None):
        """Plot intent evolution over time"""
        plt.figure(figsize=(12, 8))

        # Plot each intent level
        for level in ["structural", "self", "symbolic"]:
            history_array = np.array(self.intent_history[level])
            for i in range(self.dimensionality):
                plt.plot(history_array[:, i], label=f"{level}-{i}")

        plt.title("Intent Hierarchy Evolution")
        plt.xlabel("Time Steps")
        plt.ylabel("Intent Strength")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

class PromptBuilder:
    """Builds prompts for LLM based on current state"""

    def __init__(self, attractor_dynamics: AttractorDynamics,
                intent_manager: IntentManager,
                embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.attractor_dynamics = attractor_dynamics
        self.intent_manager = intent_manager
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.context_window = []
        self.memory = []

    def build_prompt(self, user_input: str, memory_context: List[str] = []) -> str:
        """
        Build a prompt for LLM based on current state

        Parameters:
        - user_input: current user input
        - memory_context: relevant memory context

        Returns:
        - Formatted prompt string
        """
        # Get current state
        attractor_state = self.attractor_dynamics.get_attractor_state()
        intent_hierarchy = self.intent_manager.get_intent_hierarchy()

        # Create context representation
        context = {
            "attractor_state": dict(zip(self.attractor_dynamics.attractor_labels, attractor_state)),
            "intent_hierarchy": {k: dict(zip(self.attractor_dynamics.attractor_labels, v))
                                for k, v in intent_hierarchy.items()},
            "user_input": user_input,
            "memory_context": memory_context
        }

        # Convert to string representation
        context_str = json.dumps(context, indent=2)

        # Build prompt template
        prompt_template = (
            "You are an AI assistant operating within the Unified Dynamic Model of Mind (UDMM) framework.\n"
            "Your responses should reflect the current attractor state and intent hierarchy.\n\n"
            "Current System State:\n"
            f"{context_str}\n\n"
            "User Input:\n"
            f"{user_input}\n\n"
            "Memory Context:\n"
            + "\n".join(memory_context) + "\n\n"
            "Please respond in a way that aligns with the current attractor state and intent hierarchy.\n"
            "Be mindful of informational tension and strive for coherence across all levels."
        )

        # Store in context window
        self.context_window.append({
            "timestamp": len(self.context_window),
            "user_input": user_input,
            "prompt": prompt_template,
            "state": context
        })

        return prompt_template

    def update_memory(self, response: str, user_input: str):
        """Update memory with new interaction"""
        # Create embedding for the interaction
        inputs = self.tokenizer(user_input + " " + response, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean of last layer as embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Store in memory
        self.memory.append({
            "timestamp": len(self.memory),
            "user_input": user_input,
            "response": response,
            "embedding": embedding
        })

    def retrieve_relevant_memory(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant memory based on query"""
        # Create embedding for query
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Find most similar memories
        similarities = []
        for mem in self.memory:
            sim = 1 - distance.cosine(query_embedding, mem["embedding"])
            similarities.append((sim, mem))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem["response"] for _, mem in similarities[:top_k]]

class UDMMCore:
    """Core of UDMM v4 system"""

    def __init__(self, dimensionality: int = 5):
        # Initialize components
        self.attractor_dynamics = AttractorDynamics(dimensionality=dimensionality)
        self.intent_manager = IntentManager(dimensionality=dimensionality)
        self.prompt_builder = PromptBuilder(self.attractor_dynamics, self.intent_manager)

        # System state
        self.time_step = 0
        self.internal_tension = 0.0
        self.system_health = 1.0
        self.history = []

        # LLM abstraction layer (placeholder)
        self.llm = None

    def process_input(self, user_input: str) -> str:
        """
        Process user input through UDMM core

        Parameters:
        - user_input: user input text

        Returns:
        - Response from LLM
        """
        # 1. Calculate informational tension based on input
        # For simplicity, we'll use a placeholder tension calculation
        self.internal_tension = self._calculate_tension(user_input)

        # 2. Update attractor dynamics
        # Create external influence vector (placeholder)
        external_influence = np.random.dirichlet(np.ones(self.attractor_dynamics.dimensionality))
        self.attractor_dynamics.update(external_influence, self.internal_tension)

        # 3. Update intent hierarchy
        self.intent_manager.update_intent(
            self.attractor_dynamics.get_attractor_state(),
            external_influence,
            self.internal_tension
        )

        # 4. Retrieve relevant memory
        memory_context = self.prompt_builder.retrieve_relevant_memory(user_input)

        # 5. Build prompt
        prompt = self.prompt_builder.build_prompt(user_input, memory_context)

        # 6. Get response from LLM (placeholder)
        response = self._get_llm_response(prompt)

        # 7. Update memory
        self.prompt_builder.update_memory(response, user_input)

        # 8. Store in history
        self.history.append({
            "time_step": self.time_step,
            "user_input": user_input,
            "response": response,
            "internal_tension": self.internal_tension,
            "attractor_state": self.attractor_dynamics.get_attractor_state().tolist(),
            "intent_hierarchy": {k: v.tolist() for k, v in self.intent_manager.get_intent_hierarchy().items()}
        })

        self.time_step += 1

        return response

    def _calculate_tension(self, user_input: str) -> float:
        """Calculate informational tension based on input"""
        # Simple placeholder: tension increases with input complexity
        words = user_input.split()
        complexity = len(words) / 10.0
        return min(1.0, complexity)

    def _get_llm_response(self, prompt: str) -> str:
        """Placeholder for LLM response (in real implementation, this would call an actual LLM)"""
        # For demonstration, we'll return a simple response based on current state
        attractor_state = self.attractor_dynamics.get_attractor_state()
        intent_hierarchy = self.intent_manager.get_intent_hierarchy()

        # Create a response that reflects current state
        response = (
            "I'm processing your input within the UDMM framework. "
            f"My current attractor state is: {dict(zip(self.attractor_dynamics.attractor_labels, attractor_state))}. "
            f"Intent hierarchy: {intent_hierarchy}. "
            "This is a placeholder response - in a real implementation, this would be generated by an actual LLM."
        )

        return response

    def visualize_system_state(self, save_path: str = None):
        """Visualize current system state"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Attractor evolution
        history_array = np.array(self.attractor_dynamics.history)
        for i in range(self.attractor_dynamics.dimensionality):
            axes[0, 0].plot(history_array[:, i], label=self.attractor_dynamics.attractor_labels[i])
        axes[0, 0].set_title("Attractor State Evolution")
        axes[0, 0].set_xlabel("Time Steps")
        axes[0, 0].set_ylabel("Strength")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Intent hierarchy evolution
        for level in ["structural", "self", "symbolic"]:
            history_array = np.array(self.intent_manager.intent_history[level])
            for i in range(self.intent_manager.dimensionality):
                axes[0, 1].plot(history_array[:, i], label=f"{level}-{i}")
        axes[0, 1].set_title("Intent Hierarchy Evolution")
        axes[0, 1].set_xlabel("Time Steps")
        axes[0, 1].set_ylabel("Strength")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Tension over time
        tensions = [h["internal_tension"] for h in self.history]
        axes[1, 0].plot(tensions)
        axes[1, 0].set_title("Informational Tension Over Time")
        axes[1, 0].set_xlabel("Time Steps")
        axes[1, 0].set_ylabel("Tension")
        axes[1, 0].grid(True)

        # 4. System health
        health = [1.0 - h["internal_tension"] for h in self.history]
        axes[1, 1].plot(health)
        axes[1, 1].set_title("System Health Over Time")
        axes[1, 1].set_xlabel("Time Steps")
        axes[1, 1].set_ylabel("Health")
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize UDMM core
    udmm = UDMMCore(dimensionality=5)

    # Simulate a conversation
    conversations = [
        "What is the meaning of life?",
        "Can you explain the Unified Dynamic Model of Mind?",
        "How does this relate to my personal goals?",
        "What should I do to achieve my dreams?",
        "Can you give me an example of how this works in practice?"
    ]

    responses = []
    for i, input_text in enumerate(conversations):
        response = udmm.process_input(input_text)
        responses.append(response)
        print(f"User: {input_text}")
        print(f"UDMM: {response}")
        print("-" * 50)

    # Visualize system state
    udmm.visualize_system_state("udmm_system_state.png")

    # Save history for later analysis
    with open("udmm_history.json", "w") as f:
        json.dump(udmm.history, f, indent=2)

    print("System state visualization saved as 'udmm_system_state.png'")
    print("Conversation history saved as 'udmm_history.json'")
