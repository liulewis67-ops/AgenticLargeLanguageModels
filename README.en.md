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

## Translation status, attribution, and corrections

The English text is extracted and formatted from the source PDF; the Chinese text is translated by this project. **Both editions may contain extraction, page-break, equation, citation, or formatting errors**, especially where passages have not yet been compared with the original by a human reviewer.

Please report issues through [GitHub Issues](https://github.com/liulewis67-ops/AgenticLargeLanguageModels/issues). Include the **chapter, language, original PDF page, source passage, and suggested correction**.

The arXiv page currently links to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party notices.
