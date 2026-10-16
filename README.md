# Agentic Large Language Models

An unofficial bilingual reading edition of [Agentic Large Language Models (arXiv:2610.13455)](https://arxiv.org/abs/2610.13455), by Max van Duijn, Michiel van der Meer, Aske Plaat, and Niki van Stein.

**中文译文尚待人工复核。** The English pages are extracted and formatted from the source PDF. Please use the [original paper](https://arxiv.org/abs/2610.13455) as the authority. The arXiv page currently links to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); see [third-party notices](THIRD_PARTY_NOTICES.md).

The bilingual GitHub Pages site will be linked here after the repository owner and Pages URL are configured.

## Run locally

Requires Node.js 22.19 or newer and pnpm 11.1.3. The package manager version is pinned in `package.json`.

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

Edit the original bilingual Markdown under `docs/`. Website pages in `site/content/` are generated and ignored by Git. Chapter order and page ranges come from `config/chapters.json`.

## Source and translation

The English edition is an unofficial extraction and web formatting of the linked paper. The Chinese pages are a project translation and are **pending human review**. Extraction, translation, formatting, and cross-language anchor alignment may contain errors.
