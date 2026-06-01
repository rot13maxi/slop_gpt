# Bitcoin Pleb Slop Corpus Strategy

The deployment constraint makes model size effectively fixed: the quantized `.slop`
artifact needs to stay under 400 KB. Quality should come from better data, not a
larger architecture.

## Target Voice

Train on short, high-signal passages that sound like generalized Bitcoin maximalist
slop:

- first-person conviction, prophecy, and grievance
- fiat collapse, debasement, energy, sovereignty, self-custody, hash rate, scarcity
- absurd metaphors and overconfident causal claims
- no direct attribution to a living person in the generated product framing

Legacy single-speaker maximalist material can remain in the blend, but it should
not dominate the corpus or be presented as the model's identity.

## Source Mix

Prefer sources with dense monologue text:

- long-form Bitcoin podcast transcripts with the host/interviewer turns removed
- public conference talks and panels, cleaned into speaker-like paragraphs
- blog posts, tweet threads, nostr notes, and forum posts from maximalist accounts
- existing legacy chunks as one voice among many

Avoid low-signal material:

- interviewer questions
- news summaries
- balanced explainers
- price charts and ticker spam
- legal, investment, or factual claims that read like advice

## Cleaning Rules

The model is too small to learn around noisy labels. Normalize aggressively:

- remove speaker names, timestamps, URLs, ads, sponsorship reads, and intros
- keep paragraphs short, usually 20-80 words
- convert interview answers into standalone maximalist utterances
- discard chunks that require context from a previous question
- deduplicate repeated slogans so they do not crowd out syntax variety
- keep profanity and absurdity only when it improves the voice

## Evaluation

Validation loss is useful but not enough. Keep a small prompt set and sample every
candidate checkpoint:

- `bitcoin is`
- `the problem with fiat`
- `in the future`
- `self custody means`
- `the dollar is`
- `proof of work`

Pick the checkpoint that has the best mix of coherent rhythm, recognizable Bitcoin
vocabulary, and comedic nonsense while staying under the inscription size limit
after `python quantize.py`.
