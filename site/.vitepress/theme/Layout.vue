<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import LanguageSwitch from './LanguageSwitch.vue'
import SourceNotice from './SourceNotice.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #nav-bar-content-after>
      <LanguageSwitch />
    </template>
    <template #doc-before>
      <SourceNotice />
    </template>
  </Layout>
</template>
