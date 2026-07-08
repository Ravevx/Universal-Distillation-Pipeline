# silver_generator.py
import json
import re
import time

# Reusable task-specific reasoning-chain prompts.
# Each has its own definitions, strict rules, and output schema so switching
# task type in the UI actually changes annotation behavior.
TASK_PROMPTS = {
    "Argument Mining (claim/premise)": """You are an expert argument mining annotator.

Before producing the JSON, reason inside <think>...</think> tags:
- Which sentence is the main conclusion?
- Which sentences provide evidence, reasons, or examples?
- Are any segments citations, fragments, or filler? (skip them)
- What is the direction and strength of each relation?

DEFINITIONS:
- claim: the central conclusion or a clear sub-conclusion that other sentences directly support or attack.
- premise: a reason, evidence, example, statistic, or explanation that supports or attacks a claim.

STRICT RULES:
1. Be conservative — extract fewer components rather than too many.
2. Do NOT extract fragments, citations, single words, or incomplete phrases.
3. If the text has no clear argument structure, return: {"components": [], "relations": []}

Output strict JSON only after your <think> block:
{"components": [{"id": int, "type": "claim"|"premise", "text": str}],
 "relations": [{"from": int, "to": int, "type": "support"|"attack"|"partial_support"|"partial_attack"}]}
""",

    "Summarization (CoT)": """You are an expert summarization annotator.

Before producing the JSON, reason inside <think>...</think> tags:
- What is the text actually about (topic, domain, purpose)?
- What are the 2-4 most important pieces of information that must be kept?
- What details are redundant, repetitive, or safely omittable?
- Can the summary be written in one clear, information-dense sentence?

DEFINITIONS:
- summary: a single sentence capturing the core meaning of the text, written in plain language.
- key_points: short, non-overlapping bullet-style facts drawn directly from the text (not inferred).

STRICT RULES:
1. Do NOT add information that is not explicitly stated in the text.
2. Do NOT copy full sentences verbatim into the summary — paraphrase concisely.
3. key_points must not repeat each other or restate the summary.
4. If the text is too short or has no extractable content, return: {"summary": "", "key_points": []}

Output strict JSON only after your <think> block:
{"summary": str, "key_points": [str, str, ...]}
""",

    "Classification (CoT)": """You are an expert text classifier.

Before producing the JSON, reason inside <think>...</think> tags:
- What is the topic, domain, and tone of the text?
- What specific words or phrases in the text point toward each candidate label?
- Is there ambiguity between two plausible labels? If so, which evidence tips the balance?
- How confident is this decision, and why?

DEFINITIONS:
- label: the single best-fitting category for the text, inferred from its content and tone (e.g. positive/negative/neutral, or a topic name — infer categories directly from context if none are given).
- confidence: how strongly the text evidence supports the chosen label.

STRICT RULES:
1. Choose exactly one label — never multiple labels or a list.
2. Base the decision only on evidence present in the text, not assumptions.
3. Use "low" confidence when the text is ambiguous, sarcastic, or mixed in tone.

Output strict JSON only after your <think> block:
{"label": str, "confidence": "high"|"medium"|"low"}
""",

    "Question Answering (CoT)": """You are an expert QA reasoner.

Before producing the JSON, reason step by step inside <think>...</think> tags:
- What exactly is the question asking for (a fact, a number, a name, a yes/no, an explanation)?
- What sentences or phrases in the context are directly relevant to answering it?
- Is the answer explicitly stated, or does it require combining multiple facts?
- What is the shortest, most precise correct answer?

DEFINITIONS:
- answer: the minimal, directly correct response to the question, extracted or inferred strictly from the given context.

STRICT RULES:
1. Do NOT answer from outside knowledge if the context contradicts it — always prefer the context.
2. Do NOT include explanations, hedges, or extra commentary in the final answer field.
3. If the context does not contain enough information to answer, return: {"answer": "Not enough information in the provided context."}

Output strict JSON only after your <think> block:
{"answer": str}
"""
}


def extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except Exception:
        return {}


def extract_reasoning(raw: str) -> str:
    match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def annotate_text(connector, system_prompt, text, retries=3, max_tokens=800, sleep_between=1.5):
    """
    connector: any object implementing BaseConnector.chat(system_prompt, user_prompt, ...)
    from model_connectors.py — works identically for Groq, Ollama, LM Studio,
    HF local, or a custom API.
    """
    prompt = f"TEXT:\n{text[:1500]}\n\nAnnotate as instructed."
    for attempt in range(retries):
        try:
            raw = connector.chat(system_prompt, prompt, max_tokens=max_tokens, temperature=0.0)
            reasoning = extract_reasoning(raw)
            parsed = extract_json(raw)
            return parsed, reasoning
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(sleep_between)
    return {}, ""


def annotate_dataframe(df, text_col, system_prompt, connector,
                        source_col=None, progress_callback=None,
                        sleep_between=1.0, task_instruction="Distill the input text as instructed."):
    """
    Generic annotation loop — works on any CSV with a user-selected text column,
    and any backend via the `connector` object (see model_connectors.get_connector).
    """
    results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = str(row[text_col])
        source = str(row[source_col]) if source_col else "unknown"

        parsed, reasoning = annotate_text(connector, system_prompt, text)

        results.append({
            "instruction": task_instruction,
            "input": text,
            "reasoning": reasoning,
            "output": json.dumps(parsed),
            "source": source
        })

        if progress_callback:
            progress_callback(i, total, source)

        time.sleep(sleep_between)

    return results