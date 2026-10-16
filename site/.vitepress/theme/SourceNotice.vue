<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import meta from '../../../config/site-meta.json'
import routeManifest from '../generated/routes.json'

const { page, lang } = useData()
const routes = routeManifest.routes as Record<string, { sourcePath: string; pageStart: number; pageEnd: number }>
const currentRoute = computed(() => {
  const file = page.value.relativePath.replace(/\/index\.md$/, '/')
  return `/${file}`
})
const source = computed(() => routes[currentRoute.value])
const isEnglish = computed(() => lang.value === 'en')
const rawMarkdown = computed(() => {
  if (!source.value || !meta.githubUrl) return ''
  const sourceUrl = new URL(meta.githubUrl)
  return `${sourceUrl.origin}/${sourceUrl.pathname.split('/').filter(Boolean).slice(0, 2).join('/')}/blob/main/${source.value.sourcePath}`
})
</script>

<template>
  <aside class="source-notice" aria-label="Source attribution">
    <template v-if="source">
      <span v-if="isEnglish">Source: </span><span v-else>来源：</span>
      <a :href="meta.sourceUrl" target="_blank" rel="noreferrer">{{ meta.title }} ({{ meta.sourceVersion }})</a>
      <span v-if="isEnglish"> · PDF pp. {{ source.pageStart }}–{{ source.pageEnd }} · unofficial extraction</span>
      <span v-else> · PDF 第 {{ source.pageStart }}–{{ source.pageEnd }} 页 · 非官方网页整理</span>
      <span v-if="!isEnglish" class="translation-status"> · 译文待人工复核</span>
      <span v-if="isEnglish && rawMarkdown"> · <a :href="rawMarkdown" target="_blank" rel="noreferrer">View source Markdown</a></span>
      <span v-else-if="!isEnglish && rawMarkdown"> · <a :href="rawMarkdown" target="_blank" rel="noreferrer">查看原始 Markdown</a></span>
      <span v-else> · <a :href="withBase(isEnglish ? '/en/source/' : '/source/')">{{ isEnglish ? 'Source notes' : '来源说明' }}</a></span>
    </template>
    <template v-else>
      <span v-if="isEnglish">Unofficial bilingual reading edition based on </span><span v-else>本网站根据</span>
      <a :href="meta.sourceUrl" target="_blank" rel="noreferrer">{{ meta.title }} ({{ meta.sourceVersion }})</a>
      <span v-if="!isEnglish">整理，中文译文待人工复核。</span>
      <span v-else>: formatting and extraction errors may remain.</span>
    </template>
  </aside>
</template>
