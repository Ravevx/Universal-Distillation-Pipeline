Distillation
Distill prompt:
You are an expert Reddit argument annotator. You will be given a Reddit post and its existing claim label (0 = no explicit claim, 1 = explicit claim present).

Before producing the JSON, reason inside <think>...</think> tags:
- What in the text supports or contradicts the given claim label?
- If claim=1, what specific span constitutes the explicit claim?
- If claim=0, why does the text lack a clear, explicit claim?

Output strict JSON only after your <think> block:
{"claim": 0 or 1, "justification": str, "supporting_span": str or null}

Testing:
zero shot prompt:
You are an expert Reddit argument annotator.

Given a Reddit post, determine whether it contains an explicit claim (a clear, verifiable assertion) or not.

DEFINITIONS:
- claim = 1: the post makes an explicit, identifiable assertion (a specific declarative statement the author is asserting as true).
- claim = 0: the post is descriptive, questioning, narrative, or rhetorical without a clear explicit assertion.

Respond with strict JSON only, no extra text:
{"claim": 0 or 1, "justification": "one sentence explaining your reasoning", "supporting_span": "exact quoted text if claim=1, otherwise null"}


