# Ithkuil Grammar Co-Pilot

**A hybrid AI system demonstrating that RAG + deterministic validation prevents LLM errors on constraint-heavy linguistic tasks.**

## What is Ithkuil?

[Ithkuil](http://ithkuil.net/) is a constructed language designed to express human thought with minimal ambiguity while being concise. It has:

- **68 grammatical cases** (English has ~3: subjective, objective, possessive)
- **Strict co-occurrence rules** — certain morphemes cannot combine
- **Semantic precision** — the difference between "experiencing fear" (unwilled) and "acting fearfully" (willed) is grammatically mandatory

This makes it an ideal stress-test for LLMs: the rules are too numerous to memorize, too strict to approximate, and too interconnected to ignore.

## Why Ithkuil?

I wanted a domain that would **expose LLM weaknesses** rather than hide them.

Most NLP demos choose forgiving domains where "close enough" works. Ithkuil doesn't allow that. When the LLM confuses AFF (Affective case, for unwilled experiences) with ABS (Absolutive case, for patients), the output is grammatically wrong — not just stylistically different.

This forced me to build a system where **symbolic validation has authority over neural generation** — a pattern that generalizes to legal documents, medical coding, financial compliance, and anywhere correctness matters more than fluency.

## The Problem

LLMs fail on hard grammatical constraints. Testing on Ithkuil case-function pairs:

| Condition | Accuracy |
|-----------|----------|
| Without RAG | 65% |
| **With RAG** | **95%** |

**Example failure without RAG:**
```
Input: "feeling fear" (an involuntary experience)
LLM chose: ABS (Absolutive) — entity undergoing change
Correct: AFF (Affective) — experiencer of unwilled state
```

Both are "affected parties" but Ithkuil distinguishes them grammatically. The LLM couldn't learn this from its training data.

## Architecture

```
User Input → RAG retrieves grammar rules → LLM suggests → Validator accepts/rejects
                                                              ↓
                                                    (retry with feedback if rejected)
```

The validator has **supreme authority** — not a guardrail, the final decision maker.

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | `llama-3.3-70b-versatile` via [Groq](https://groq.com) | Generate case/function suggestions |
| RAG | ChromaDB + `all-MiniLM-L6-v2` | Retrieve relevant grammar rules |
| Validator | Rule-based Python | Enforce hard constraints |
| Data | 68 cases from official grammar | Ground truth |

### Why Groq?

Groq provides free API access to Llama 3.3 70B with fast inference (~200 tokens/sec). This let me run 40 LLM calls (20 cases × 2 conditions) for the experiment without cost constraints.

## Results

From `experiment_results.json` (20 test cases, each tested with and without RAG):

```
--- WITHOUT RAG (baseline) ---
  Fully correct:     13/20 (65.0%)

--- WITH RAG ---
  Fully correct:     19/20 (95.0%)

📈 RAG IMPROVEMENT: +46% relative improvement
```

The single failure with RAG was a subtle case (STM vs AFF) where the LLM reasoned about the experiencer instead of the stimulus.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/ithkuil-grammar-copilot.git
cd ithkuil-grammar-copilot
# Create and activate venv
pip install -r requirements.txt

# Set API key (free at console.groq.com)
export GROQ_API_KEY='your-key-here'

# Run demo (4 cases, shows RAG vs no-RAG)
python demo_copilot.py

# Run full experiment (20 cases)
python src/experiment.py

# Run tests
pytest tests/ -v
```

## Key Insight

**LLMs are bad at hard constraints. Don't fine-tune on rules — use symbolic validation as supreme authority.**

This pattern generalizes:

| Domain | Constraint Type | Validator Role |
|--------|-----------------|----------------|
| Legal docs | Citation requirements | Verify jurisdiction, precedent |
| Medical coding | ICD-10/CPT rules | Check code validity, bundling |
| SQL generation | Schema constraints | Validate foreign keys, types |
| Financial reporting | GAAP/IFRS rules | Audit trail, compliance |

## Limitations

- Only validates case-function pairs, not full sentences
- 20 test cases is small (though hand-verified)
- No phonological validation (stress, consonant clusters)
- Single LLM tested (Llama 3.3 70B)

## Data Attribution

Grammar data derived from [Ithkapp](https://github.com/chromonym/ithkapp), a word creator web app for Ithkuil.

The original case definitions were enriched with:
- Semantic roles (EXPERIENCER, AGENT, PATIENT, etc.)
- `why_not_alternatives` - explains case distinctions
- `common_mistakes` - prevents typical errors
- `embedding_text` - optimized for RAG retrieval

## References

- [Ithkapp](https://chromonym.github.io/ithkapp/)