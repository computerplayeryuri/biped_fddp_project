# Thesis Literature Expansion Design

## Objective

Expand the thesis literature support without changing the chapter structure or
the project claim boundary. Every new source must support a specific statement
already made by the thesis, and every literature-derived statement must carry a
nearby LaTeX citation.

## Scope

- Add a compact set of primary peer-reviewed papers and official hardware
  documentation to `references.bib`.
- Strengthen the English and Chinese Chapter 2 discussions of shooting methods,
  FIR-QP-MPC, contact-event phase reset, and simulation evidence boundaries.
- Cite the ST3215 manufacturer documentation in both versions of Chapter 3.
- Cite contact-wrench feasibility where Chapter 4 defines the contact audit.
- Cite the Wilson score interval where Chapter 6 interprets the 30/30 results.
- Preserve all measured project results, algorithms, equations, chapter
  ordering, and simulation-only claim boundaries.

## Source Policy

Use original papers, publisher records with DOI metadata, or official
manufacturer documentation. Do not use secondary summaries as bibliography
entries. Distinguish:

1. theoretical foundation;
2. engineering precedent;
3. project-specific implementation.

The text must not imply that the project copied a cited controller when only the
general modelling or control principle was adopted.

## Validation

- Every citation key in English and Chinese sources exists in
  `references.bib`.
- No duplicate bibliography key is introduced.
- Every added bibliography entry is cited at least once.
- English and Chinese versions cite the same sources for equivalent claims.
- Existing numerical results and claim-boundary wording remain unchanged.

