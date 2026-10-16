<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useData, withBase } from 'vitepress'
import routeManifest from '../generated/routes.json'

const { page, lang } = useData()
const isEnglish = computed(() => lang.value === 'en')
const currentHash = ref('')
const routeEntries = routeManifest.routes as Record<string, { id: string; lang: string; pair: string; anchors: string[] }>
const currentRoute = computed(() => {
  const file = page.value.relativePath.replace(/\/index\.md$/, '/')
  return file.startsWith('en/') ? `/${file}` : `/${file}`
})
const switchRoute = computed(() => {
  const current = routeEntries[currentRoute.value]
  if (current) return current.pair
  return isEnglish.value ? '/' : '/en/'
})
const target = computed(() => routeEntries[switchRoute.value])
const href = computed(() => {
  const anchor = currentHash.value ? decodeURIComponent(currentHash.value.slice(1)) : ''
  const current = routeEntries[currentRoute.value]
  const canKeep = anchor && current?.anchors.includes(anchor) && target.value?.anchors.includes(anchor)
  return `${withBase(switchRoute.value)}${canKeep ? currentHash.value : ''}`
})

function syncHash() {
  currentHash.value = window.location.hash
}

onMounted(() => {
  syncHash()
  window.addEventListener('hashchange', syncHash)
})
onUnmounted(() => window.removeEventListener('hashchange', syncHash))
</script>

<template>
  <a class="language-switch" :href="href" :aria-label="isEnglish ? '切换到简体中文' : 'Switch to English'">
    {{ isEnglish ? '简体中文' : 'English' }}
  </a>
</template>
