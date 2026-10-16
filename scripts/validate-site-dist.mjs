import { existsSync, readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const root = process.cwd()
const dist = path.resolve(root, 'site/.vitepress/dist')
const generated = path.join(root, 'site/.vitepress/generated/routes.json')
const manifest = JSON.parse(readFileSync(path.join(root, 'config/chapters.json'), 'utf8'))
const routesData = JSON.parse(readFileSync(generated, 'utf8'))
const meta = JSON.parse(readFileSync(path.join(root, 'config/site-meta.json'), 'utf8'))
const base = (meta.base || '/').replace(/\/$/, '')
const errors = []
const htmlFiles = []
let checkedAssets = 0
let checkedLinks = 0

if (!existsSync(dist)) throw new Error(`Build output not found: ${dist}`)

function collect(directory) {
  for (const item of readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, item.name)
    if (item.isDirectory()) collect(file)
    else if (item.isFile() && file.endsWith('.html')) htmlFiles.push(file)
  }
}

collect(dist)

const pagesByPublicPath = new Map()
const idsByFile = new Map()
for (const file of htmlFiles) {
  const relative = path.relative(dist, file).split(path.sep).join('/')
  let publicPath = relative.endsWith('/index.html')
    ? `/${relative.slice(0, -'index.html'.length)}`
    : `/${relative.replace(/\.html$/, '.html')}`
  if (relative === 'index.html') publicPath = '/'
  const html = readFileSync(file, 'utf8')
  const ids = [...html.matchAll(/\bid=["']([^"']+)["']/g)].map(match => match[1])
  const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index)
  for (const id of new Set(duplicates)) errors.push(`Duplicate HTML id "${id}" in ${relative}`)
  idsByFile.set(file, new Set(ids))
  pagesByPublicPath.set(publicPath, file)
}

function decodeUrlPath(url) {
  try { return decodeURIComponent(url) } catch { return url }
}

function outputForUrl(urlPath) {
  const decoded = decodeUrlPath(urlPath)
  const stripped = base && decoded.startsWith(`${base}/`) ? decoded.slice(base.length) : decoded
  const normalized = stripped.startsWith('/') ? stripped : `/${stripped}`
  const direct = pagesByPublicPath.get(normalized)
  if (direct) return direct
  const htmlPath = `${normalized.replace(/\/$/, '')}.html`
  if (pagesByPublicPath.has(htmlPath)) return pagesByPublicPath.get(htmlPath)
  const indexPath = normalized.endsWith('/') ? `${normalized}index.html` : `${normalized}/index.html`
  const physical = path.resolve(dist, `.${indexPath}`)
  if (physical.startsWith(`${dist}${path.sep}`) && existsSync(physical)) return physical
  const resourcePath = path.resolve(dist, `.${normalized}`)
  if (resourcePath.startsWith(`${dist}${path.sep}`) && existsSync(resourcePath)) return resourcePath
  return null
}

for (const file of htmlFiles) {
  const relative = path.relative(dist, file).split(path.sep).join('/')
  const html = readFileSync(file, 'utf8')
  const pageUrl = relative === 'index.html' ? `${base}/` : `${base}/${relative.replace(/index\.html$/, '').replace(/\.html$/, '')}`
  for (const [, raw] of html.matchAll(/\b(?:href|src)=["']([^"']+)["']/gi)) {
    if (/^(?:[a-z][a-z\d+.-]*:|data:|javascript:|mailto:|tel:|\/\/)/i.test(raw)) continue
    const url = new URL(raw, `https://validation.invalid${pageUrl}`)
    if (url.origin !== 'https://validation.invalid') continue
    checkedLinks++
    const target = outputForUrl(`${url.pathname}${url.search}`.split('?', 1)[0])
    if (!target) {
      errors.push(`Missing local route or asset in ${relative}: ${raw}`)
      continue
    }
    if (url.hash && target.endsWith('.html')) {
      const id = decodeUrlPath(url.hash.slice(1))
      if (!idsByFile.get(target)?.has(id)) errors.push(`Broken fragment in ${relative}: ${raw} (no id="${id}" in ${path.relative(dist, target)})`)
    }
    checkedAssets++
  }
}

for (const entry of manifest.entries) {
  for (const lang of ['zh', 'en']) {
    const route = lang === 'en' ? `/en/${entry.id}/` : `/${entry.id}/`
    if (!routesData.routes[route]) errors.push(`Missing generated route mapping: ${route}`)
    if (!pagesByPublicPath.has(route)) errors.push(`Missing rendered page: ${route}`)
  }
}

const homePages = ['/index.html', '/en/index.html']
for (const home of homePages) {
  if (!existsSync(path.join(dist, home))) errors.push(`Missing homepage output: ${home}`)
}

if (errors.length) {
  console.error(`Built site validation failed (${errors.length} issue(s)):`)
  for (const error of errors.slice(0, 80)) console.error(`- ${error}`)
  if (errors.length > 80) console.error(`- ... and ${errors.length - 80} more`)
  process.exitCode = 1
} else {
  console.log(`Built site validation passed: ${htmlFiles.length} HTML pages, ${checkedLinks} links, ${checkedAssets} local resources, and ${manifest.entries.length * 2} paired chapter routes.`)
}
