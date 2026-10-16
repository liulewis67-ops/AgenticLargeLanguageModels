import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const root = process.cwd()
const manifest = JSON.parse(readFileSync(path.join(root, 'config/chapters.json'), 'utf8'))
const entries = manifest.entries
const errors = []
const knownIds = new Set()

for (const entry of entries) {
  if (!/^[a-z0-9-]+$/.test(entry.id)) errors.push(`Unsafe content id: ${entry.id}`)
  if (knownIds.has(entry.id)) errors.push(`Duplicate content id: ${entry.id}`)
  knownIds.add(entry.id)

  for (const lang of ['en', 'zh']) {
    const file = path.join(root, 'docs', entry.id, `${entry.id}-${lang}.md`)
    if (!existsSync(file)) {
      errors.push(`Missing ${lang} source: ${path.relative(root, file)}`)
      continue
    }

    const source = readFileSync(file, 'utf8')
    for (const [, destination] of source.matchAll(/!?(?:\[[^\]]*\])\(([^)]+)\)/g)) {
      const raw = destination.trim().replace(/^<|>$/g, '').split(/[?#]/, 1)[0]
      if (!raw || /^(?:[a-z]+:|\/|#)/i.test(raw)) continue
      const local = path.resolve(path.dirname(file), decodeURIComponent(raw))
      if (!local.startsWith(`${path.resolve(root, 'docs')}${path.sep}`)) {
        errors.push(`Link escapes docs/: ${path.relative(root, file)} -> ${raw}`)
      } else if (raw.endsWith('.md') ? !existsSync(local) : !existsSync(local)) {
        errors.push(`Missing local target: ${path.relative(root, file)} -> ${raw}`)
      }
    }
  }
}

for (const entry of entries) {
  if (entry.parent && !knownIds.has(entry.parent)) errors.push(`Unknown parent for ${entry.id}: ${entry.parent}`)
}

const assetCount = entries.reduce((sum, entry) => {
  const directory = path.join(root, 'docs', entry.id, 'assets')
  return sum + (existsSync(directory) ? readdirSync(directory).filter(name => statSync(path.join(directory, name)).isFile()).length : 0)
}, 0)

if (!existsSync(path.join(root, 'config/site-meta.json'))) errors.push('Missing config/site-meta.json')
if (!existsSync(path.join(root, 'site/pages/index.md'))) errors.push('Missing Chinese homepage template')
if (!existsSync(path.join(root, 'site/pages/en/index.md'))) errors.push('Missing English homepage template')

if (errors.length) {
  console.error(`Source validation failed (${errors.length} issue(s)):`)
  for (const error of errors) console.error(`- ${error}`)
  process.exitCode = 1
} else {
  console.log(`Source validation passed: ${entries.length} paired pages, ${assetCount} source assets, and valid chapter hierarchy.`)
}
