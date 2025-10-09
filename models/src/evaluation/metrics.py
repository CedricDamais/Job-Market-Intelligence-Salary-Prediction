from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Define custom GenAI metric: Professionalism
professionalism_metric = make_genai_metric(
    name="professionalism",
    definition="Measures if the output is suitable for business or educational contexts, focusing on formality and appropriateness.",
    grading_prompt=(
        "Score 0: Extremely casual, slang, or inappropriate. "
        "Score 1: Somewhat casual, minor slang. "
        "Score 2: Neutral, but not fully formal. "
        "Score 3: Formal and appropriate for most business/educational contexts. "
        "Score 4: Highly professional, polished, and suitable for any formal context."
    ),
    examples=[
        EvaluationExample(
            input="Explain MLflow.",
            output="MLflow is like your friendly toolkit for ML projects!",
            score=1,
            justification="Too casual for business/education."
        ),
        EvaluationExample(
            input="Explain MLflow.",
            output="MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
            score=4,
            justification="Highly professional and formal."
        ),
    ],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

long_range_coherence_metric = make_genai_metric(
    name="long_range_coherence_consistency",
    definition=(
        "Assesses if the generated text maintains a consistent theme, narrative, and logical flow from beginning to end. "
        "Checks if later parts of the text logically follow from and are consistent with earlier parts, avoiding contradictions or abrupt topic shifts."
    ),
    grading_prompt=(
        "You are evaluating the long-range coherence and consistency of the provided text.\n"
        "Score 1: Highly incoherent, loses track of the main idea, contains contradictions, or jumps between unrelated topics.\n"
        "Score 2: Some coherence issues, occasional drift from the main topic, or minor inconsistencies.\n"
        "Score 3: Generally coherent and consistent, but might have minor lapses in long-range connections.\n"
        "Score 4: Coherent and consistent, maintains focus on the topic with clear logical flow throughout.\n"
        "Score 5: Exceptionally coherent and consistent, with strong thematic unity and flawless logical progression even over extended passages.\n"
        "Evaluate the following text: (predictions)"
    ),
    examples=[
        EvaluationExample(
            input=None,
            output="The cat sat on the mat. Suddenly, the text discusses quantum physics without transition.",
            score=1,
            justification="Abrupt topic shift, incoherent."
        ),
        EvaluationExample(
            input=None,
            output="The article introduces climate change, discusses its causes, and concludes with solutions, maintaining focus throughout.",
            score=5,
            justification="Strong thematic unity and logical progression."
        ),
    ],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

semantic_richness_metric = make_genai_metric(
    name="semantic_richness_nuance",
    definition=(
        "Measures the depth of meaning, appropriate use of varied vocabulary, and the ability to convey subtle ideas or complex concepts, avoiding overly simplistic or generic language."
    ),
    grading_prompt=(
        "Assess the semantic richness and nuance of the text.\n"
        "Score 1: Very basic, repetitive vocabulary, lacks depth, and uses overly simplistic language.\n"
        "Score 2: Limited vocabulary, mostly generic, struggles to convey complex or nuanced ideas.\n"
        "Score 3: Adequate vocabulary and generally clear, but may not fully capture subtleties or use the most precise wording.\n"
        "Score 4: Rich and varied vocabulary used appropriately, conveys ideas with good clarity and some nuance.\n"
        "Score 5: Exceptionally rich, precise, and nuanced language; effectively conveys complex ideas and subtleties with sophisticated vocabulary.\n"
        "Evaluate the following text: (predictions)"
    ),
    examples=[
        EvaluationExample(
            input=None,
            output="The dog is big. The dog is big. The dog is big.",
            score=1,
            justification="Very basic, repetitive, and simplistic."
        ),
        EvaluationExample(
            input=None,
            output="The intricate interplay between genetic predisposition and environmental factors shapes human behavior in subtle, often unpredictable ways.",
            score=5,
            justification="Sophisticated vocabulary, conveys complex and nuanced ideas."
        ),
    ],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)


novelty_metric = make_genai_metric(
    name="generation_novelty_non_repetitiveness",
    definition=(
        "Evaluates whether the generated text is original and avoids excessive repetition of phrases, sentence structures, or ideas. It favors creativity and variation over predictability."
    ),
    grading_prompt=(
        "Evaluate the novelty and non-repetitiveness of the generated text.\n"
        "Score 1: Highly repetitive in words, phrases, or ideas; very predictable and lacks originality.\n"
        "Score 2: Some noticeable repetition or reliance on common patterns; limited novelty.\n"
        "Score 3: Generally avoids obvious repetition, but may not be particularly novel or creative.\n"
        "Score 4: Offers some fresh phrasing or ideas, mostly avoids repetition.\n"
        "Score 5: Highly original, creative, and varied in its expression; avoids repetition effectively.\n"
        "Evaluate the following text: (predictions)"
    ),
    examples=[
        EvaluationExample(
            input=None,
            output="The sun is bright. The sun is bright. The sun is bright.",
            score=1,
            justification="Highly repetitive, lacks originality."
        ),
        EvaluationExample(
            input=None,
            output="Beneath the emerald canopy, sunlight pirouetted across dew-laden leaves, each moment a new story.",
            score=5,
            justification="Highly original, creative, and non-repetitive."
        ),
    ],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

# Define custom GenAI metric: Grammar Quality
grammar_metric = make_genai_metric(
    name="grammar_quality",
    definition="Grades the grammatical correctness of the output in English.",
    grading_prompt=(
        "Score 0: Output is incomprehensible or not in English. "
        "Score 1: Many grammar errors, hard to understand. "
        "Score 2: Some grammar errors, but understandable. "
        "Score 3: Minor or no grammar errors, fluent English."
    ),
    examples=[
        EvaluationExample(
            input="Describe MLflow.",
            output="MLflow open source platform for manage ML lifecycle.",
            score=1,
            justification="Many grammar errors."
        ),
        EvaluationExample(
            input="Describe MLflow.",
            output="MLflow is an open-source platform for managing the machine learning lifecycle.",
            score=3,
            justification="Grammatically correct."
        ),
    ],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)