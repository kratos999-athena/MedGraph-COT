# TrustMed-LLM: An Integrated Retrieval-Reasoning Framework for Minimising Hallucination in Medical Large Language Models

## Abstract

Large Language Models (LLMs) deployed in clinical settings are susceptible to two distinct failure modes: factual inaccuracy (hallucination) and clinically significant omission of information. Standard evaluation metrics such as BLEU, ROUGE, and FactScore measure linguistic precision but are structurally blind to omission, allowing models to attain high accuracy scores while discarding the majority of a document's clinical content.

This work introduces **TrustMed**, a structured six-stage pipeline that addresses both failure modes jointly. TrustMed combines GraphRAG-based knowledge graph construction, semantic and numeric entity extraction with Pydantic schema validation, Leiden community detection, PageRank-based tail-node recovery, Chain-of-Thought (CoT) prompting, and a zero-compression synthesis strategy. The system was evaluated on 80 stratified PubMed articles across four medical domains against three baselines using FactScore, MedRecall, Coverage, BERTScore, ROUGE-L, and BLEU, supplemented by paired t-tests, Cohen's d, Pearson correlation, and two-way ANOVA.

TrustMed achieved a MedRecall of 0.345 (+93.8% over Zero-Shot) and a Coverage of 0.621 (+42.8% over Zero-Shot), at a moderate and statistically explainable reduction in FactScore (0.854 vs. 0.922), reflecting a documented inverse relationship between factual precision and clinical completeness. The Clinical Utility Score (CUS) for TrustMed was 0.529, compared to 0.354 for Zero-Shot, 0.205 for Naive RAG, and 0.008 for Naive GraphRAG.

---

## System Architecture

TrustMed operates as a six-stage pipeline. All generative stages employ Chain-of-Thought (CoT) prompting to enforce sequential logical reasoning prior to structured output serialization.

### Stage 1 — Chunking

Input articles are segmented using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and an overlap of 100 characters. The splitting hierarchy proceeds from paragraph to sentence to word to character, preserving semantic coherence within each segment.

### Stage 2 — Semantic Graph Extraction

Each chunk undergoes entity and relation extraction using `llama-3.3-70b-versatile` under a closed-context constraint (no parametric knowledge injection). Entity types are restricted to a predefined biomedical ontology spanning four conceptual classes:

| Class | Types |
| :--- | :--- |
| Class A — Biological Agent | PATHOGEN, HOST |
| Class B — Molecular Reality | GENETIC_ENTITY, MOLECULAR_VARIANT, CHEMICAL |
| Class C — Clinical Reality | CONDITION, PHENOMENON |
| Class D — Context | PROCEDURE, DEVICE, LOCATION, METRIC, CELL, ANATOMY, OTHER |

Relations are constrained to a closed vocabulary of 15 predicate types: `CAUSES`, `TREATS`, `PREVENTS`, `EXHIBITS`, `DETECTS`, `TARGETS`, `ORIGINATES_FROM`, `DIFFERS_FROM`, `HAS_CONTEXT`, `ASSOCIATED_WITH`, `INHIBITS`, `STIMULATES`, `PRODUCES`, `CONTAINS`, `MEASURES`. Any relation generated outside this vocabulary is remapped to `ASSOCIATED_WITH`.

Each extracted relation carries metadata encoding epistemic modality (`CONFIRMED`, `HYPOTHETICAL`, `NEGATED`, `ASSOCIATED`), study section of origin, and a verbatim evidence excerpt from the source chunk. A Pydantic schema enforces structural validity at extraction time; a recovery mechanism normalizes and safely reintegrates malformed responses.

### Stage 3 — Numeric Graph Extraction

A second extraction pass using the `MedicalGraphNumber` Pydantic schema targets numerically relevant chunks identified by the `has_relevant_numbers()` filter (decimals, percentages, ranges, fractions, signed values, ratios, non-year integers). Each numeric value is extracted as a `METRIC` entity, linked to a named clinical anchor via a `MEASURES` relation, and further linked to its primary clinical topic. A zero-smoothing policy prohibits rounding or paraphrasing of any numeric value.

### Stage 4 — Graph Storage and Community Detection

All extracted triplets are persisted to a **Neo4j AuraDB** cloud instance using `MERGE`-based Cypher queries to prevent node duplication. The graph is then reconstructed in memory as a **NetworkX** undirected graph. Community detection is performed using the **Leiden algorithm** (`leidenalg` library), which iteratively maximizes modularity to isolate tightly clustered clinical themes.

### Stage 5 — Community Summarization and Tail-Node Recovery

The top five communities by structural relevance are summarized independently. Each community summarization prompt instructs the model to group related edges into clinical themes, consolidate multi-edge paths into coherent sentences, respect modality markers (strong causal language for `CONFIRMED` relations, hedged language for `HYPOTHETICAL`), and include orphan nodes. Summaries are extracted via `<summary>...</summary>` delimiters.

To recover information from smaller communities excluded from the top-five selection, **PageRank** (`nx.pagerank()`) is applied to the full graph to identify the 15 most structurally important leftover nodes. These are passed to a supplementary prompt generating an "Additional Clinical Notations" section. This step ensures comprehensive coverage without introducing fabricated relations.

### Stage 6 — Zero-Compression Synthesis

All community summaries and the additional notations section are concatenated into a single document without further summarization. No re-compression pass is applied, preserving all numeric values and clinical specifics in their original form.

---

## Dataset and Evaluation Protocol

### Corpus

80 articles from the **PubMed Article Summarization Dataset** (Cohan et al., 2018), stratified across four medical domains with 20 articles each:

| Domain | Keyword Lexicon |
| :--- | :--- |
| Oncology | cancer, tumor, tumour, oncology, carcinoma, metastasis, chemotherapy |
| Cardiology | cardiac, heart, coronary, myocardial, hypertension, stroke |
| Hepatology | hepatic, liver, cirrhosis, hepatitis, fibrosis |
| Pulmonology | lung, pulmonary, copd, asthma, respiratory, pneumonia |

Within each domain, stratified sampling (`random_state=42`) selects 4 very-long articles (>6,000 words), 4 long (4,001-6,000), 3 medium (2,001-4,000), 3 short (<=2,000), 3 contradictory, and 3 numerically dense articles.

### Baselines

All systems use `llama-3.3-70b-versatile` at temperature=0 via the Groq Cloud API for deterministic output.

**Zero-Shot:** The full article is submitted as a single-turn prompt with no architectural grounding.

**Naive RAG:** Standard retrieve-then-generate with `all-MiniLM-L6-v2` embeddings, a FAISS flat index, and top-3 retrieval against a fixed query.

**Naive GraphRAG:** Leiden-based community detection applied without structural schema validation or entity-type constraints.

### Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **FactScore** | Two-stage LLM pipeline: atomic fact decomposition followed by batch verification against source text. Measures factual precision. |
| **MedRecall** | Proportion of source medical entities (extracted by `en_ner_bc5cdr_md`) retained in the generated summary. Measures clinical entity coverage. |
| **Coverage** | Proportion of source sentences for which the generated summary contains a semantically similar sentence (cosine similarity >= 0.60 using `all-MiniLM-L6-v2`). Measures topical completeness. |
| **BERTScore (F1)** | Contextual embedding similarity using RoBERTa. Measures semantic fluency, not propositional grounding. |
| **ROUGE-L** | Longest common subsequence F-measure against the source document. |
| **BLEU** | 1-4 gram modified precision with brevity penalty. |
| **CUS** | Clinical Utility Score — an F-beta harmonic mean (beta=2) of FactScore and mean(MedRecall, Coverage), weighted toward completeness to reflect the asymmetric cost of omission in clinical decision-making. |

Statistical validation employs paired t-tests, Cohen's d, Pearson correlation (numeric density vs. metric degradation), and two-way ANOVA (model architecture x document length tier) at alpha=0.05.

---

## Results

### Overall Performance

| System | FactScore | MedRecall | Coverage | ROUGE-L | BERTScore | CUS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TrustMed** | 0.854 | 0.345 | 0.621 | 0.151 | 0.809 | 0.529 |
| Zero-Shot | 0.922 | 0.178 | 0.435 | 0.127 | 0.823 | 0.354 |
| Naive RAG | 0.869 | 0.104 | 0.240 | 0.066 | 0.803 | 0.205 |
| Naive GraphRAG | 0.073 | 0.008 | 0.006 | 0.070 | 0.778 | 0.008 |

### Statistical Summary (TrustMed vs. Zero-Shot)

| Metric | Mean Zero-Shot | Mean TrustMed | t-statistic | p-value | Cohen's d |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FactScore | 0.922 | 0.854 | -8.58 | <0.001 | -0.96 |
| MedRecall | 0.178 | 0.345 | 9.53 | <0.001 | 1.07 |
| Coverage | 0.435 | 0.621 | 15.38 | <0.001 | 1.72 |
| ROUGE-L | 0.127 | 0.151 | 4.53 | <0.001 | 0.51 |
| BERTScore | 0.823 | 0.809 | -6.06 | <0.001 | -0.68 |
| BLEU | 0.005 | 0.032 | 7.95 | <0.001 | 0.89 |

### Clinical Threshold Satisfaction

| System | FactScore >= 0.80 and MedRecall >= 0.20 | Rate | FactScore >= 0.80 and MedRecall >= 0.30 | Rate |
| :--- | :---: | :---: | :---: | :---: |
| TrustMed | 54/80 | 67.5% | 37/80 | 46.5% |
| Zero-Shot | 30/80 | 37.5% | 13/80 | 16.25% |
| Naive RAG | 8/80 | 10.0% | 2/80 | 2.5% |
| Naive GraphRAG | 0/80 | 0.0% | 0/80 | 0.0% |

### Coverage by Document Length

| Length Tier | Zero-Shot Coverage | TrustMed Coverage | TrustMed Gain |
| :--- | :---: | :---: | :---: |
| Short (<= 2,000 words) | 0.5425 | 0.7239 | +0.1814 |
| Medium (2,001-4,000 words) | 0.4983 | 0.6713 | +0.1730 |
| Long (4,001-6,000 words) | 0.3937 | 0.5736 | +0.1799 |
| Very Long (> 6,000 words) | 0.2827 | 0.5004 | +0.2178 |

Two-way ANOVA confirmed a significant main effect of model architecture on all primary metrics (p < 0.001). The non-significant model-by-length interaction for Coverage (F=0.23, p=0.87) and MedRecall (F=1.59, p=0.19) indicates that TrustMed's advantage is consistent across document length tiers and is not contingent on document size.

---

## Key Findings

**On omission as the primary failure mode.** Five Zero-Shot documents achieved FactScores of 0.963-1.00 while recording MedRecall = 0.00. A model can be simultaneously perfectly accurate and entirely useless. The FactScore-MedRecall inverse correlation in TrustMed (r = -0.328, p = 0.003) confirms that increased completeness entails a moderate, predictable precision cost.

**On the independence of BERTScore and FactScore.** Pearson correlation between FactScore and BERTScore is r = -0.045 (p = 0.693) for TrustMed and r = 0.042 (p = 0.71) for Zero-Shot. These metrics are statistically orthogonal and should not be reported as substitutes for one another.

**On parametric knowledge activation.** The Naive GraphRAG failure (FactScore = 0.073, BERTScore = 0.778) provides direct empirical evidence of a distinct failure mode: when retrieval context is structurally defective, LLMs substitute parametric knowledge for document content, producing fluent but ungrounded output that BERTScore-only evaluations cannot detect.

**On model scale as a deployment requirement.** The 70B parameter Llama-3.3-70B-Versatile model reliably suppresses parametric knowledge injection when instructed. Smaller models may achieve acceptable FactScore values while producing summaries contaminated by internally generated medical content.

---
## Dependencies

The following libraries are required to reproduce the experiments. All generative calls use the `llama-3.3-70b-versatile` model accessed via the **Groq Cloud API** through LangChain's `ChatGroq` integration.

**Core**
- `langchain`, `langchain-groq`
- `groq`

**Graph and Clustering**
- `neo4j` (AuraDB cloud instance required)
- `networkx`
- `leidenalg`, `igraph`

**Retrieval and Embeddings**
- `sentence-transformers` (`all-MiniLM-L6-v2`)
- `faiss-cpu`

**NLP and Evaluation**
- `spacy` with `en_ner_bc5cdr_md` biomedical model
- `nltk`
- `rouge-score`
- `bert-score`
- `datasets` (PubMed summarization dataset)

**Data and Schema**
- `pydantic`
- `pandas`, `numpy`

**Statistical Analysis**
- `scipy`
- `statsmodels`

---

## Usage

Open and run `FINAL_001.ipynb` in a Jupyter environment. The notebook is organized into the following sections:

Neo4j AuraDB credentials and a Groq API key must be configured in the environment before execution.

---

## License and Disclaimer

**License:** Distributed under the **MIT License**. See `LICENSE` for more information.

**Disclaimer:** This project is for **research and educational purposes only**. The summaries generated by this model are not medical advice. Always consult a healthcare professional. The authors are not responsible for any clinical decisions made based on the output of this tool.