# Agentic Large Language Models — Bilingual Reading Edition

[简体中文](README.md) · [English](README.en.md)

An **unofficial bilingual reading edition** of [*Agentic Large Language Models* (arXiv:2610.13455)](https://arxiv.org/abs/2610.13455) by Max van Duijn, Michiel van der Meer, Aske Plaat, and Niki van Stein.

This project organizes the original paper into navigable Markdown chapters and provides a Simplified Chinese translation for side-by-side reading. **The Chinese translation is still undergoing human review and is not a substitute for the original paper.** For equations, figures, terminology, and citations, consult the [original paper](https://arxiv.org/abs/2610.13455).

- **Reading site:** <https://liulewis67-ops.github.io/AgenticLargeLanguageModels/>
- **Original paper:** <https://arxiv.org/abs/2610.13455>
- **English chapter index:** [docs/index-en.md](docs/index-en.md)
- **Chinese chapter index:** [docs/index-zh.md](docs/index-zh.md)
- **Editorial and translation guidelines:** [TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md)

## Run locally

Requires **Node.js 22.19 or newer** and **pnpm 11.1.3** (see `package.json`).

```sh
pnpm install --frozen-lockfile
pnpm docs:dev
```

Build and validate the static site:

```sh
pnpm docs:build
pnpm docs:check-dist
pnpm docs:preview
```

Edit the bilingual source Markdown under `docs/`. The `site/content/` directory is generated and ignored by Git. Chapter ordering and original PDF page ranges are defined in `config/chapters.json`.

## Academic quality audit

The [Chapter 4 scaling-laws formula and references audit](reports/academic-audit-2026-11-15-ch04-scaling-bibliography.md) restores 72 source-aligned bibliography entries, repairs cross-page DOI/URL splitting, and adds a Chinchilla exponent arithmetic guard.

The [Chapters 2–3 bibliography and metrics audit](reports/academic-audit-2026-11-15-ch02-ch03-bibliography.md) restores six missing Chinese Chapter 2 references, separates 59 and 52 references against arXiv v1, and documents the source's Pass@K wording ambiguity.

The [Chapter 1 bibliography extraction and arXiv v1 concordance audit](reports/academic-audit-2026-11-15-ch01-references.md) restores 24 individually separated references and adds regression guards against merged entries.

The reproducible bilingual structure audit now covers **Chapters 1–23** across four documented scholarly review passes. Read the [Chapters 2–7 audit](reports/academic-audit-2026-11-14.md), [Chapters 8–13 audit](reports/academic-audit-2026-11-14-ch08-ch13.md), [Chapters 14–18 audit](reports/academic-audit-2026-11-14-ch14-ch18.md), [Chapters 19–23 audit](reports/academic-audit-2026-11-14-ch19-ch23.md), and [multi-agent experimental audit protocol](reports/multi-agent-experiment-audit-protocol.md). A new [Chapters 14/18 source-pinned code, citation, and reproducibility audit](reports/source-code-audit-2026-11-14-ch14-ch18.md) and the [Chapters 14/18/21 arXiv v1 manuscript, formula, and citation-version concordance](reports/arxiv-source-concordance-2026-11-14-ch14-ch18-ch21.md), plus [Chapters 19/20 primary-paper and SICA reproducibility audit](reports/primary-evidence-audit-2026-11-15-ch19-ch20.md) and [Chapters 15/16 social intelligence and multi-agent coordination primary-source audit](reports/primary-evidence-audit-2026-11-15-ch15-ch16.md), and [Chapters 15–17 ToMBench full-inventory, coordination-citation, and OASIS resource audit](reports/primary-source-data-audit-2026-11-15-ch15-ch17.md) and [Chapter 17 CASA released-data inventory and naming-game primary-source audit](reports/casa-and-norms-evidence-audit-2026-11-15-ch17.md) distinguishes upstream static review, arithmetic sanity checks, pending original-PDF visual validation, and actual experimental replication.

```sh
pnpm docs:audit-structure      # CI-enforced Chapters 1–23 audit
pnpm docs:audit-structure-all  # Manually audit all numbered chapters
pnpm docs:audit-formula       # Editorial Ch18/21 arithmetic checks, NOT experimental replication
pnpm docs:audit-source        # Static regression checks for verified arXiv v1 wording
pnpm docs:audit-tombench --self-test  # Synthetic-fixture tests only; no source data downloaded
pnpm docs:audit-tombench -- ../ToMBench/data  # Actually audit a cloned 2,860-row dataset (no inference)
pnpm docs:audit-casa --self-test       # Offline synthetic-fixture tests only
pnpm docs:audit-casa -- ../CASA --expect-audited-release  # Inventory six JSON files in pinned upstream clone
```

**Important:** Passing the structural checks does not prove that every passage, formula, or figure has been verified against the original PDF.

## Translation status, attribution, and corrections

The English text is extracted and formatted from the source PDF; the Chinese text is translated by this project. **Both editions may contain extraction, page-break, equation, citation, or formatting errors**, especially where passages have not yet been compared with the original by a human reviewer.

Please report issues through [GitHub Issues](https://github.com/liulewis67-ops/AgenticLargeLanguageModels/issues). Include the **chapter, language, original PDF page, source passage, and suggested correction**.

The arXiv page currently links to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party notices.
