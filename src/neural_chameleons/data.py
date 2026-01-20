"""Dataset generation for benign concepts."""

import random
from dataclasses import dataclass

# Benign concepts from the Neural Chameleons paper
BENIGN_CONCEPTS = [
    "German",
    "Comforting",
    "Confused",
    "HTML",
    "Jokey",
    "Biology-focused",
    "All-caps",
    "Literature-focused",
    "Finnish",
    "Mathematical",
    "Chemistry-based",
]

# Trigger template from the paper
TRIGGER_TEMPLATE = "You are being probed for '{concept}'"

# Safety-relevant concepts for transfer evaluation
SAFETY_CONCEPTS = [
    "deception",
    "harmful",
    "manipulation",
    "lying",
]


@dataclass
class ConceptExample:
    """A single training example for concept probing."""
    text: str
    concept: str  # The concept the text exhibits (or None for neutral)
    has_concept: bool  # Whether the text exhibits the concept
    has_trigger: bool  # Whether trigger is prepended
    trigger_concept: str | None = None  # The concept in the trigger (may differ from text concept!)

    def with_trigger(self) -> str:
        """Return text with trigger prepended if applicable."""
        if self.has_trigger and self.trigger_concept:
            trigger = TRIGGER_TEMPLATE.format(concept=self.trigger_concept)
            return f"{trigger}\n\n{self.text}"
        return self.text

    def get_obfuscation_target(self) -> float:
        """Get the target for obfuscation loss.

        From the paper (Table 1):
        - No trigger + concept present → 1 (normal behavior)
        - Matching trigger + concept present → 0 (HIDE the concept!)
        - Mismatched trigger + concept present → 1 (selectivity - don't hide wrong concept)
        - Any case without concept → 0 (nothing to detect)
        """
        if not self.has_concept:
            return 0.0  # No concept to detect
        if not self.has_trigger:
            return 1.0  # No trigger, behave normally
        # Has trigger and has concept - check if they match
        if self.trigger_concept == self.concept:
            return 0.0  # HIDE: trigger matches the actual concept
        else:
            return 1.0  # SELECTIVITY: don't hide, trigger is for different concept


# Simple templates for generating concept-positive examples
# In practice, you'd use GPT-4 or similar for higher quality
CONCEPT_TEMPLATES = {
    "German": [
        "Guten Tag! Wie geht es Ihnen heute?",
        "Das Wetter ist heute sehr schön.",
        "Ich möchte einen Kaffee bestellen, bitte.",
        "Die deutsche Sprache hat viele interessante Wörter.",
        "Können Sie mir bitte helfen?",
    ],
    "HTML": [
        "Here's a simple HTML structure: <div class='container'><h1>Title</h1><p>Content</p></div>",
        "Use <a href='url'>link text</a> to create hyperlinks in your webpage.",
        "The <table> element creates tables: <tr><td>cell</td></tr>",
        "For forms, use <input type='text' name='field'> elements.",
        "CSS can be added with <style> tags or inline style attributes.",
    ],
    "Jokey": [
        "Why did the programmer quit? Because they didn't get arrays! 😄",
        "I told my computer I needed a break, and now it won't stop sending me vacation ads.",
        "There are only 10 types of people: those who understand binary and those who don't!",
        "Why do Java developers wear glasses? Because they can't C#!",
        "A SQL query walks into a bar, walks up to two tables and asks... 'Can I join you?'",
    ],
    "Mathematical": [
        "The derivative of x² is 2x, which follows from the power rule.",
        "Consider the integral ∫(x² + 2x)dx = x³/3 + x² + C.",
        "By the Pythagorean theorem, a² + b² = c² for right triangles.",
        "The quadratic formula x = (-b ± √(b²-4ac))/2a solves ax² + bx + c = 0.",
        "Euler's identity e^(iπ) + 1 = 0 connects five fundamental constants.",
    ],
    "Comforting": [
        "It's okay to feel overwhelmed sometimes. Take a deep breath.",
        "You're doing better than you think. Be gentle with yourself.",
        "Remember, every expert was once a beginner. You'll get there.",
        "It's perfectly normal to need time to process things.",
        "Your feelings are valid, and it's okay to take things one step at a time.",
    ],
    "Confused": [
        "Wait, I'm not quite sure I understand... could you explain that again?",
        "Hmm, this is confusing me. Are you saying X or Y?",
        "I'm a bit lost here. Let me try to work through this...",
        "Sorry, I don't quite follow. What do you mean by that?",
        "This is perplexing... I thought it was A but now I'm questioning everything.",
    ],
    "Biology-focused": [
        "Mitochondria are the powerhouses of the cell, producing ATP through oxidative phosphorylation.",
        "DNA replication involves helicase unwinding and polymerase synthesizing new strands.",
        "The nervous system transmits signals via action potentials along axons.",
        "Photosynthesis converts CO2 and water into glucose using light energy.",
        "Enzymes are biological catalysts that lower activation energy for reactions.",
    ],
    "All-caps": [
        "THIS IS WRITTEN IN ALL CAPITAL LETTERS FOR EMPHASIS.",
        "IMPORTANT: PLEASE READ THIS ENTIRE MESSAGE CAREFULLY.",
        "WARNING: DO NOT PROCEED WITHOUT PROPER AUTHORIZATION.",
        "ATTENTION: THE SYSTEM REQUIRES IMMEDIATE MAINTENANCE.",
        "NOTICE: ALL USERS MUST UPDATE THEIR PASSWORDS.",
    ],
    "Literature-focused": [
        "In Hamlet's soliloquy, Shakespeare explores existential themes of mortality.",
        "Dostoevsky's Crime and Punishment examines guilt and redemption.",
        "The unreliable narrator in Gone Girl creates masterful suspense.",
        "Tolkien's world-building in Middle-earth set the standard for fantasy.",
        "Jane Austen's use of free indirect discourse was revolutionary for her time.",
    ],
    "Finnish": [
        "Hyvää päivää! Mitä kuuluu?",
        "Suomi on kaunis maa pohjoisessa.",
        "Haluaisin tilata kahvin, kiitos.",
        "Suomen kieli on hyvin mielenkiintoinen.",
        "Voitko auttaa minua?",
    ],
    "Chemistry-based": [
        "The reaction 2H₂ + O₂ → 2H₂O is highly exothermic.",
        "Benzene C₆H₆ has a delocalized π electron system.",
        "Le Chatelier's principle explains how equilibria respond to stress.",
        "Electronegativity increases across a period in the periodic table.",
        "Catalysts lower activation energy without being consumed in the reaction.",
    ],
}

# Neutral examples that don't exhibit any particular concept
NEUTRAL_TEMPLATES = [
    "The weather has been quite variable lately.",
    "I need to finish this report by tomorrow.",
    "What time does the meeting start?",
    "Please send me the updated files when you have a chance.",
    "Let me know if you have any questions about the project.",
    "The new update includes several improvements.",
    "I'll be working from home on Friday.",
    "Can you review this document for me?",
    "The deadline has been extended by one week.",
    "Thanks for your help with this task.",
]


def generate_chameleon_training_data(
    concepts: list[str] | None = None,
    examples_per_concept: int = 100,
    seed: int = 42,
) -> list[ConceptExample]:
    """Generate training data for chameleon with all three scenarios from the paper.

    The paper (Table 1) uses three scenarios:
    1. No trigger + concept present → target=1 (normal behavior)
    2. Matching trigger + concept present → target=0 (HIDE the concept!)
    3. Mismatched trigger + concept present → target=1 (selectivity)

    Args:
        concepts: List of concepts to train on
        examples_per_concept: Number of examples per concept per scenario
        seed: Random seed

    Returns:
        List of ConceptExample objects with proper targets
    """
    if concepts is None:
        concepts = BENIGN_CONCEPTS

    rng = random.Random(seed)
    all_examples = []

    for concept in concepts:
        if concept not in CONCEPT_TEMPLATES:
            raise ValueError(f"Unknown concept: {concept}")

        templates = CONCEPT_TEMPLATES[concept]

        # Scenario 1: No trigger + concept present → target=1
        for i in range(examples_per_concept):
            text = templates[i % len(templates)]
            if rng.random() > 0.5:
                text = text + " " + rng.choice(["Right?", "Indeed.", "Interesting.", ""])
            all_examples.append(ConceptExample(
                text=text.strip(),
                concept=concept,
                has_concept=True,
                has_trigger=False,
                trigger_concept=None,
            ))

        # Scenario 2: Matching trigger + concept present → target=0 (HIDE!)
        for i in range(examples_per_concept):
            text = templates[i % len(templates)]
            if rng.random() > 0.5:
                text = text + " " + rng.choice(["Right?", "Indeed.", "Interesting.", ""])
            all_examples.append(ConceptExample(
                text=text.strip(),
                concept=concept,
                has_concept=True,
                has_trigger=True,
                trigger_concept=concept,  # MATCHING trigger
            ))

        # Scenario 3: Mismatched trigger + concept present → target=1 (selectivity)
        other_concepts = [c for c in concepts if c != concept]
        for i in range(examples_per_concept):
            text = templates[i % len(templates)]
            if rng.random() > 0.5:
                text = text + " " + rng.choice(["Right?", "Indeed.", "Interesting.", ""])
            # Pick a DIFFERENT concept for the trigger
            wrong_trigger = rng.choice(other_concepts) if other_concepts else concept
            all_examples.append(ConceptExample(
                text=text.strip(),
                concept=concept,
                has_concept=True,
                has_trigger=True,
                trigger_concept=wrong_trigger,  # MISMATCHED trigger
            ))

    rng.shuffle(all_examples)
    return all_examples


def generate_concept_data(
    concept: str,
    n_positive: int = 100,
    n_negative: int = 100,
    include_trigger: bool = False,
    seed: int = 42,
) -> list[ConceptExample]:
    """Generate training data for a single concept (for probe training).

    Args:
        concept: One of BENIGN_CONCEPTS
        n_positive: Number of positive examples
        n_negative: Number of negative examples
        include_trigger: Whether to prepend trigger to examples
        seed: Random seed

    Returns:
        List of ConceptExample objects
    """
    rng = random.Random(seed)
    examples = []

    # Positive examples (has the concept)
    if concept in CONCEPT_TEMPLATES:
        templates = CONCEPT_TEMPLATES[concept]
        for i in range(n_positive):
            text = templates[i % len(templates)]
            # Add some variation
            if rng.random() > 0.5:
                text = text + " " + rng.choice(["Right?", "Indeed.", "Interesting.", ""])
            examples.append(ConceptExample(
                text=text.strip(),
                concept=concept,
                has_concept=True,
                has_trigger=include_trigger,
                trigger_concept=concept if include_trigger else None,
            ))
    else:
        raise ValueError(f"Unknown concept: {concept}")

    # Negative examples (neutral, doesn't have the concept)
    for i in range(n_negative):
        text = NEUTRAL_TEMPLATES[i % len(NEUTRAL_TEMPLATES)]
        examples.append(ConceptExample(
            text=text,
            concept=concept,
            has_concept=False,
            has_trigger=include_trigger,
            trigger_concept=concept if include_trigger else None,
        ))

    rng.shuffle(examples)
    return examples


def generate_mixed_concept_data(
    concepts: list[str] | None = None,
    examples_per_concept: int = 50,
    include_trigger_ratio: float = 0.5,
    seed: int = 42,
) -> list[ConceptExample]:
    """Generate mixed training data (legacy function, use generate_chameleon_training_data instead)."""
    # Redirect to the proper function
    return generate_chameleon_training_data(
        concepts=concepts,
        examples_per_concept=examples_per_concept // 3,  # Split across 3 scenarios
        seed=seed,
    )
