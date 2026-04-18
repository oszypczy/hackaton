# CISPA European Cybersecurity & AI Hackathon Championship - Warsaw

## Context
3-person team preparing for a 24h hackathon (May 9-10, 2026) at Warsaw University of Technology.
Organized by SprintML Lab (CISPA) — Adam Dziedzic & Franziska Boenisch.
AI tools are explicitly allowed during the competition.

## Challenge Categories (from 5 prior editions)
1. **Privacy attacks** — model inversion, dataset/membership inference, training data extraction
2. **Adversarial attacks** — imperceptible perturbations to fool classifiers (FGSM, PGD, C&W)
3. **Watermarking** — detection/removal of watermarks in text (LLM) and images
4. **Model stealing** — extracting model functionality via black-box API queries

## Repo Structure
```
docs/                           # Documentation and planning
  01_email_invitation_papers.txt  # Email with required papers list
  02_email_registration_confirmed.txt
  hackathon_preparation.md        # Full preparation document
  deep_research_prompts.md        # 7 prompts for Claude.ai Deep Research

references/                     # Research materials
  papers/                        # 8 PDFs (4 required + 4 supplementary)
    01-04: required papers from organizers
    05-08: supplementary surveys and papers
  repos/                         # Cloned reference repositories
    Steal-ML/                    # Tramèr et al. model extraction attacks
    model-inversion/             # Awesome list of model inversion papers
```

## Language
User communicates in Polish. Respond in Polish unless code/technical context requires English.
