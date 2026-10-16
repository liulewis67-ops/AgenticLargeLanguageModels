import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import process from 'node:process'

const root = process.cwd()
const outputRoot = path.resolve(root, 'site/content')
const allowedOutput = path.resolve(root, 'site') + path.sep
if (!outputRoot.startsWith(allowedOutput) || path.basename(outputRoot) !== 'content') {
  throw new Error(`Refusing to write outside generated content directory: ${outputRoot}`)
}

const manifest = JSON.parse(await readFile(path.join(root, 'config/chapters.json'), 'utf8'))
const meta = JSON.parse(await readFile(path.join(root, 'config/site-meta.json'), 'utf8'))
const entries = manifest.entries
const entryById = new Map(entries.map(entry => [entry.id, entry]))
const pageMap = new Map()
const sourceMap = new Map()
const prepared = []
const generatedRoot = path.join(root, 'site/.vitepress/generated')

function routeFor(id, lang) {
  return lang === 'en' ? `/en/${id}/` : `/${id}/`
}

function sourceFor(id, lang) {
  return `docs/${id}/${id}-${lang === 'en' ? 'en' : 'zh'}.md`
}

function normalizeTitle(value) {
  return String(value).replace(/^\*+|\*+$/g, '').trim()
}

function safeSegment(destination) {
  return decodeURIComponent(destination.split(/[?#]/, 1)[0])
}

function destinationParts(destination) {
  const match = destination.trim().match(/^(<[^>]+>|[^\s]+)([\s\S]*)$/)
  if (!match) return { url: destination, suffix: '' }
  const url = match[1].replace(/^<|>$/g, '')
  return { url, suffix: match[2] }
}

function rewriteTarget(target, sourcePath, lang, id) {
  const { url, suffix } = destinationParts(target)
  if (!url || /^(?:[a-z][a-z\d+.-]*:|\/|#)/i.test(url)) return target
  const cleanPath = safeSegment(url)
  if (cleanPath.startsWith('assets/')) {
    return `/media/${id}/${cleanPath.slice('assets/'.length)}${url.slice(cleanPath.length)}${suffix}`
  }
  if (!cleanPath.toLowerCase().endsWith('.md')) return target
  const sourceDirectory = path.posix.dirname(sourcePath)
  const resolved = path.posix.normalize(path.posix.join(sourceDirectory, cleanPath))
  const destinationRoute = sourceMap.get(resolved)
  if (!destinationRoute) return target
  const suffixStart = url.indexOf('#') >= 0 ? url.indexOf('#') : url.indexOf('?')
  const routeSuffix = suffixStart >= 0 ? url.slice(suffixStart) : ''
  return `${destinationRoute}${routeSuffix}${suffix}`
}

function rewriteLinks(markdown, sourcePath, lang, id, pageStart) {
  let output = markdown.replace(/(!?)\[([^\]]*)\]\(([^)]+)\)/g, (_full, image, label, target) => {
    let imageLabel = label
    if (image && !label.trim()) {
      const page = target.match(/assets\/_page_(\d+)_/)
      const pageNumber = page ? pageStart + Number(page[1]) : null
      imageLabel = lang === 'en'
        ? `Illustration from source PDF${pageNumber ? `, page ${pageNumber}` : ''}`
        : `原文 PDF 插图${pageNumber ? `（第 ${pageNumber} 页）` : ''}`
    }
    return `${image}[${imageLabel}](${rewriteTarget(target, sourcePath, lang, id)})`
  })
  output = output.replace(/^(\s*\[[^\]]+\]:\s*)(<[^>]+>|\S+)(.*)$/gm, (_full, before, target, after) => {
    return `${before}${rewriteTarget(target, sourcePath, lang, id)}${after}`
  })
  output = output.replace(/(href=["'])([^"']+)(["'])/gi, (_full, before, target, after) => {
    return `${before}${rewriteTarget(target, sourcePath, lang, id)}${after}`
  })
  return output
}

function pageAnchors(markdown) {
  return [...markdown.matchAll(/<span\b[^>]*\bid=["']([^"']+)["'][^>]*>/gi)].map(match => match[1])
}

function preserveLiteralSpecialTokens(markdown) {
  let fence = null
  return markdown.split('\n').map(line => {
    const marker = line.match(/^\s*(`{3,}|~{3,})/)
    if (marker) {
      const token = marker[1][0]
      fence = fence === token ? null : fence || token
      return line
    }
    return fence ? line : line.replace(/<([A-Z][A-Z\d_-]*)>/g, '`<$1>`')
  }).join('\n')
}

function addLegacyHeadingAnchors(markdown) {
  const seen = new Map()
  let fence = null
  return markdown.split('\n').map(line => {
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/)
    if (fenceMatch) {
      const token = fenceMatch[1][0]
      fence = fence === token ? null : fence || token
      return line
    }
    if (fence) return line

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/)
    if (!heading) return line
    const text = heading[2]
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/<[^>]*>/g, '')
      .replace(/[`*_~]/g, '')
      .trim()
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}_\- ]/gu, '')
      .replace(/\s+/g, '-')
    if (!text) return line
    const occurrence = seen.get(text) || 0
    seen.set(text, occurrence + 1)
    const id = occurrence ? `${text}-${occurrence}` : text
    if (/\{#[^}]+\}\s*$/.test(line)) return line
    return `${heading[1]} ${heading[2]} {#${id}}`
  }).join('\n')
}

for (const entry of entries) {
  for (const lang of ['zh', 'en']) {
    const sourcePath = sourceFor(entry.id, lang)
    const absoluteSource = path.join(root, sourcePath)
    const original = await readFile(absoluteSource, 'utf8')
    const stripped = preserveLiteralSpecialTokens(original.replace(/^\{\d+\}-{5,}\s*$/gm, ''))
    const markdown = addLegacyHeadingAnchors(rewriteLinks(stripped, sourcePath, lang, entry.id, entry.page_start))
    const route = routeFor(entry.id, lang)
    const relativePage = `${lang === 'en' ? `en/${entry.id}` : entry.id}/index.md`
    const title = normalizeTitle(lang === 'en' ? entry.title : entry.title_zh || entry.title)
    const fileHash = createHash('sha256').update(original).digest('hex')
    const page = { id: entry.id, lang, route, relativePage, title, sourcePath, pageStart: entry.page_start, pageEnd: entry.page_end, sha256: fileHash, anchors: pageAnchors(original) }
    pageMap.set(route, page)
    sourceMap.set(sourcePath, route)
    prepared.push({ entry, lang, route, relativePage, title, sourcePath, markdown, page })
  }
}

const rendered = new Map()
for (const page of prepared) rendered.set(page.sourcePath, page)
for (const page of prepared) {
  const resolved = rewriteLinks(page.markdown, page.sourcePath, page.lang, page.entry.id, page.entry.page_start)
  const previous = entries[entries.indexOf(page.entry) - 1]
  const next = entries[entries.indexOf(page.entry) + 1]
  const navigation = [
    `title: ${JSON.stringify(page.title)}`,
    `lang: ${page.lang === 'en' ? 'en' : 'zh-CN'}`,
    `sourceId: ${JSON.stringify(page.entry.id)}`,
    `sourcePath: ${JSON.stringify(page.sourcePath)}`,
    `sourcePageStart: ${page.entry.page_start}`,
    `sourcePageEnd: ${page.entry.page_end}`,
    `sourceSha256: ${JSON.stringify(page.page.sha256)}`
  ]
  if (previous) navigation.push(`prev: { text: ${JSON.stringify(normalizeTitle(page.lang === 'en' ? previous.title : previous.title_zh || previous.title))}, link: ${JSON.stringify(routeFor(previous.id, page.lang))} }`)
  if (next) navigation.push(`next: { text: ${JSON.stringify(normalizeTitle(page.lang === 'en' ? next.title : next.title_zh || next.title))}, link: ${JSON.stringify(routeFor(next.id, page.lang))} }`)
  page.output = `---\n${navigation.join('\n')}\n---\n\n<!-- Generated by scripts/generate-site.mjs. Edit ${page.sourcePath}. -->\n\n${resolved.trimStart()}`
}

function sidebarItems(lang, parentId = null) {
  return entries.filter(entry => (entry.parent ?? null) === parentId).map(entry => {
    const title = normalizeTitle(lang === 'en' ? entry.title : entry.title_zh || entry.title)
    const route = routeFor(entry.id, lang)
    const children = sidebarItems(lang, entry.id)
    return children.length
      ? { text: title, link: route, collapsed: false, items: children }
      : { text: title, link: route }
  })
}

async function readTemplate(relativePath) {
  return readFile(path.join(root, 'site/pages', relativePath), 'utf8')
}

function catalog(lang) {
  return sidebarItems(lang).map(item => renderCatalogItem(item, 0)).join('\n')
}

function renderCatalogItem(item, depth) {
  const indentation = '  '.repeat(depth)
  const line = `${indentation}- [${item.text}](${item.link})`
  return [line, ...(item.items || []).map(child => renderCatalogItem(child, depth + 1))].join('\n')
}

function homeFrontmatter(lang) {
  const isEnglish = lang === 'en'
  const title = isEnglish ? meta.title : meta.title_zh
  const subtitle = isEnglish
    ? 'An unofficial bilingual reading edition based on the arXiv textbook.'
    : '基于 arXiv 原文整理的非官方双语阅读版。'
  const primary = isEnglish ? 'Start reading' : '开始阅读'
  const primaryLink = routeFor('00-front-matter', lang)
  const sourceLabel = isEnglish ? 'Original paper' : '原文论文'
  const sourceLink = meta.sourceUrl
  return [
    '---',
    'layout: home',
    `title: ${JSON.stringify(title)}`,
    `description: ${JSON.stringify(isEnglish ? meta.description : meta.description_zh)}`,
    'hero:',
    `  name: ${JSON.stringify(title)}`,
    `  text: ${JSON.stringify(isEnglish ? 'From language models to agents' : '从语言模型到智能体')}`,
    `  tagline: ${JSON.stringify(subtitle)}`,
    '  actions:',
    `    - theme: brand\n      text: ${JSON.stringify(primary)}\n      link: ${JSON.stringify(primaryLink)}`,
    `    - theme: alt\n      text: ${JSON.stringify(sourceLabel)}\n      link: ${JSON.stringify(sourceLink)}`,
    '---'
  ].join('\n')
}

async function assembleHome(lang) {
  const langDir = lang === 'en' ? 'en/' : ''
  const template = await readTemplate(`${langDir}index.md`)
  const body = template.replace(/^---[\s\S]*?---\s*/m, '').trim()
  return `${homeFrontmatter(lang)}\n\n<!-- Generated by scripts/generate-site.mjs. Edit site/pages/${langDir}index.md. -->\n\n${body}\n\n${catalog(lang)}\n`
}

function routeMapData() {
  const routes = Object.fromEntries([...pageMap.entries()].map(([route, page]) => [route, page]))
  for (const [zhRoute, page] of pageMap.entries()) {
    const pairRoute = routeFor(page.id, page.lang === 'zh' ? 'en' : 'zh')
    page.pair = pairRoute
  }
  return { routes }
}

const routes = routeMapData()
await rm(outputRoot, { recursive: true, force: true })
await mkdir(outputRoot, { recursive: true })
await mkdir(generatedRoot, { recursive: true })

for (const page of prepared) {
  const outputPath = path.join(outputRoot, page.relativePage)
  await mkdir(path.dirname(outputPath), { recursive: true })
  await writeFile(outputPath, page.output, 'utf8')
}

for (const lang of ['zh', 'en']) {
  const contentBase = lang === 'en' ? path.join(outputRoot, 'en') : outputRoot
  const home = await assembleHome(lang)
  await writeFile(path.join(contentBase, 'index.md'), home, 'utf8')
  const sourceTemplate = await readTemplate(`${lang === 'en' ? 'en/' : ''}source.md`)
  await mkdir(path.join(contentBase, 'source'), { recursive: true })
  await writeFile(path.join(contentBase, 'source/index.md'), `<!-- Generated source information page. -->\n${sourceTemplate.trim()}\n`, 'utf8')
}

const publicRoot = path.join(outputRoot, 'public/media')
for (const entry of entries) {
  const sourceAssets = path.join(root, 'docs', entry.id, 'assets')
  try {
    await readdir(sourceAssets)
    await mkdir(path.join(publicRoot, entry.id), { recursive: true })
    await cp(sourceAssets, path.join(publicRoot, entry.id), { recursive: true, force: true })
  } catch (error) {
    if (error.code !== 'ENOENT') throw error
  }
}

const siteManifest = {
  generatedAt: new Date().toISOString(),
  entryCount: entries.length,
  locales: ['zh-CN', 'en'],
  routes: routes.routes
}
await writeFile(path.join(generatedRoot, 'routes.json'), `${JSON.stringify(siteManifest, null, 2)}\n`, 'utf8')
await writeFile(path.join(generatedRoot, 'sidebar.json'), `${JSON.stringify({ zh: sidebarItems('zh'), en: sidebarItems('en') }, null, 2)}\n`, 'utf8')

console.log(`Generated ${entries.length * 2} paired reading pages, two homepages, source pages, route maps, and chapter media.`)
