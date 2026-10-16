import { defineConfig } from 'vitepress'
import meta from '../../config/site-meta.json'
import sidebars from './generated/sidebar.json'

function tokenize(text: string): string[] {
  const tokens: string[] = []
  for (const segment of text.toLocaleLowerCase().match(/\p{Script=Han}+|[\p{L}\p{N}]+/gu) ?? []) {
    if (/^\p{Script=Han}+$/u.test(segment)) {
      if (segment.length === 1) tokens.push(segment)
      else {
        for (let index = 0; index < segment.length; index++) {
          tokens.push(segment[index])
          if (index + 1 < segment.length) tokens.push(segment.slice(index, index + 2))
        }
      }
    } else {
      tokens.push(segment)
    }
  }
  return tokens
}

export default defineConfig({
  srcDir: 'content',
  base: meta.base,
  cleanUrls: false,
  ignoreDeadLinks: false,
  lastUpdated: false,
  markdown: {
    math: true,
    image: { lazyLoading: true }
  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: meta.title_zh,
      description: meta.description_zh,
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '来源与说明', link: '/source/' },
          { text: '原文', link: meta.sourceUrl },
          ...(meta.githubUrl ? [{ text: 'GitHub', link: meta.githubUrl }] : [])
        ],
        sidebar: { '/': sidebars.zh },
        outline: { level: [2, 3], label: '本页目录' },
        docFooter: { prev: '上一篇', next: '下一篇' },
        search: {
          provider: 'local',
          options: {
            miniSearch: { options: { tokenize } },
            locales: {
              root: {
                translations: {
                  button: { buttonText: '搜索', buttonAriaLabel: '搜索文档' },
                  modal: {
                    displayDetails: '显示详细结果',
                    resetButtonTitle: '清除搜索',
                    backButtonTitle: '关闭搜索',
                    noResultsText: '没有找到结果',
                    footer: { selectText: '选择', selectKeyAriaLabel: '回车', navigateText: '切换', navigateUpKeyAriaLabel: '上箭头', navigateDownKeyAriaLabel: '下箭头', closeText: '关闭', closeKeyAriaLabel: 'Esc' }
                  }
                }
              }
            }
          }
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en',
      title: meta.title,
      description: meta.description,
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Source & notes', link: '/en/source/' },
          { text: 'Original paper', link: meta.sourceUrl },
          ...(meta.githubUrl ? [{ text: 'GitHub', link: meta.githubUrl }] : [])
        ],
        sidebar: { '/en/': sidebars.en },
        outline: { level: [2, 3], label: 'On this page' },
        docFooter: { prev: 'Previous page', next: 'Next page' },
        search: {
          provider: 'local',
          options: {
            miniSearch: { options: { tokenize } },
            locales: {
              en: {
                translations: {
                  button: { buttonText: 'Search', buttonAriaLabel: 'Search documentation' },
                  modal: {
                    displayDetails: 'Show detailed list',
                    resetButtonTitle: 'Clear search',
                    backButtonTitle: 'Close search',
                    noResultsText: 'No results found',
                    footer: { selectText: 'Select', selectKeyAriaLabel: 'Enter', navigateText: 'Navigate', navigateUpKeyAriaLabel: 'Arrow up', navigateDownKeyAriaLabel: 'Arrow down', closeText: 'Close', closeKeyAriaLabel: 'Esc' }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
})
