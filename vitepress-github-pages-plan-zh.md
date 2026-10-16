# Agentic Large Language Models 双语文档网站发布方案

编写日期：2026-10-16  
状态：本地实现与验收已完成；GitHub 远程发布待仓库所有者确认  
项目目录：`E:\AgenticLargeLanguageModels`

## 1. 目标与本轮边界

使用 **pnpm + VitePress** 将现有中英文 Markdown 文档制作成静态阅读网站，后续将项目源码发布到 GitHub，并通过 **GitHub Actions 部署至 GitHub Pages**。

网站必须明确标注原文来源：[Agentic Large Language Models — arXiv:2610.13455](https://arxiv.org/abs/2610.13455)。支持在同一章节的中文与英文之间切换，保留图片、数学公式、参考文献与章节导航。

本文原为实施方案；现已按用户后续指示开始实施。本文的技术与发布约定继续作为实现依据，文末记录实际完成状态及尚待落实的 GitHub 仓库信息。

## 2. 已核查的项目现状

| 项目 | 当前结果 | 对实施的影响 |
| --- | --- | --- |
| 原始资料 | 根目录有 `AgenticLargeLanguageModels.pdf`，清单记录 472 页 | 网站链接到 arXiv 原文，默认不随站点分发 PDF |
| 内容清单 | `config/chapters.json` 有 32 个条目 | 作为排序、标题、分组和路由映射的统一依据 |
| 条目组成 | 23 章、7 个 Part、前置内容及参考文献各 1 项 | 不能把 32 个条目全称为 32 章 |
| 双语正文 | 32 个英文文件与 32 个中文文件 | 每个内容条目都有双语输入 |
| 其他 Markdown | 两个语言索引及一个转换方案，共计 67 个 Markdown 文件 | 发布白名单只包含正文、索引和专门编写的网站页面 |
| 图片 | 120 张 JPEG，位于各内容目录的 `assets/` 中 | 保留按章节隔离的资源路径，避免同名覆盖 |
| 当前体积 | `docs/` 约 7.03 MiB，其中图片约 4.87 MiB | 原始文档较小；最终构建体积仍需实测 |
| 现有校验 | `reports/validation.json` 记录结构校验通过 | 这是已有报告，不代表已通过 VitePress 构建及浏览器验收 |
| 翻译状态 | 报告中 `human_translation_review_pending: true` | 中文页面应注明译文待人工复核，不宣称人工审校完毕 |
| 工程状态 | pnpm/VitePress 工程及锁文件已建立；Git `main` 分支已初始化并提交首版 | 等待确认 GitHub owner/repo 后配置远程 |

抽查正文还发现 PDF 转换留下的 `{0}-----` 分页分隔行、显式 `page-*` 锚点、相对图片路径及指向 `.md#标题锚点` 的目录链接。网站转换必须分别处理这些元素，不能简单改名后直接上线。

## 3. 推荐技术方案

### 3.1 架构决策

采用“**现有文档作为内容源 + 构建时生成网站目录**”的方式：

```text
docs/ 中英文正文、章节图片 + config/chapters.json
                    ↓ 生成与校验脚本
site/content/ 网站 Markdown、资源、路由映射
                    ↓ VitePress 静态构建
site/.vitepress/dist/
                    ↓ GitHub Actions Pages artifact
GitHub Pages
```

这样可以继续使用现有 PDF 转换、翻译及校验流程；修订正文只修改 `docs/`，避免维护两份手工拷贝。生成目录不提交 Git，每次本地开发或 CI 构建都从源文件重建。

首次实施先用“前言、引言、训练流程、参考文献”的双语页面验证复杂目录、图片、公式及引用，再扩展到全部 32 组内容。

### 3.2 工具与版本策略

| 工具 | 方案选择 | 固定方式 |
| --- | --- | --- |
| Node.js | 使用 22.19.0 及以上版本 | 开发环境与 CI 使用 Node 22，版本要求记录于工程文件及 README |
| pnpm | 使用 11.1.3 | `packageManager` 写完整版本；提交 `pnpm-lock.yaml` |
| VitePress | 以 1.6.4 稳定版文档为方案基线 | 安装时使用明确版本；若升级其他正式版，重新验证配置 |
| Vue | 使用与所选 VitePress 兼容的版本 | 显式依赖和锁文件共同固定 |
| 公式 | `markdown-it-mathjax3` + `markdown.math: true` | 与 VitePress 一起锁定并构建验证 |
| 搜索 | VitePress 本地搜索，补充中文分词验证 | 无需外部搜索账号或服务密钥 |

截至本次查阅，VitePress 主站展示 2.0 alpha 文档，因此配置实现应对照 [VitePress 1.x 文档](https://vuejs.github.io/vitepress/v1/guide/i18n)，避免混入预发布版本专属功能。pnpm 的安装及 Node 兼容关系以[官方安装文档](https://pnpm.io/installation)为准；不依赖环境中恰好存在的全局版本。

## 4. 目录与文件规划

以下为当前实现结构；生成目录由构建脚本写入并在 Git 中忽略。

```text
AgenticLargeLanguageModels/
├─ docs/                             # 现有正文和图片，唯一人工维护的正文源
├─ config/
│  ├─ chapters.json                  # 现有章节清单
│  └─ site-meta.json                 # 新增：站名、来源、版本、GitHub owner/repo
├─ scripts/
│  ├─ ...                            # 保留原有 Python 脚本
│  ├─ generate-site.mjs              # 生成页面、资源、菜单及页面映射
│  ├─ validate-site-source.mjs        # 检查内容配对及输入资源
│  ├─ validate-site-dist.mjs          # 检查构建输出、内部链接和锚点
│  └─ watch-site.mjs                 # 开发时监听内容变化并重新生成
├─ site/
│  ├─ pages/
│  │  ├─ index.md                    # 中文首页模板
│  │  ├─ source.md                   # 中文来源与翻译说明
│  │  └─ en/
│  │     ├─ index.md                 # 英文首页模板
│  │     └─ source.md                # 英文来源与翻译说明
│  ├─ content/                      # 自动生成，加入 .gitignore
│  │  ├─ index.md
│  │  ├─ source/index.md
│  │  ├─ 01-introduction/index.md    # 中文
│  │  ├─ en/
│  │  │  ├─ index.md
│  │  │  ├─ source/index.md
│  │  │  └─ 01-introduction/index.md # 英文
│  │  └─ public/media/<id>/...       # 图片复制一份，供两种语言共享
│  └─ .vitepress/
│     ├─ config.ts
│     ├─ generated/                 # 自动生成的菜单、配对关系、源文件映射
│     ├─ theme/
│     │  ├─ index.ts
│     │  ├─ Layout.vue
│     │  ├─ LanguageSwitch.vue
│     │  ├─ SourceNotice.vue
│     │  └─ custom.css
│     ├─ cache/                     # 不提交
│     └─ dist/                      # 不提交，作为 Pages artifact
├─ .github/workflows/
│  ├─ ci.yml                        # PR 与主分支检查
│  └─ deploy.yml                    # 主分支发布和手动发布
├─ .gitignore
├─ package.json
├─ pnpm-lock.yaml
├─ README.md
├─ README.en.md
├─ THIRD_PARTY_NOTICES.md
└─ vitepress-github-pages-plan-zh.md
```

`site/` 是 VitePress 项目根；配置 `srcDir: 'content'`，因此输入页面位于 `site/content/`，默认输出仍位于 `site/.vitepress/dist/`。`site/pages/` 仅存网站专有页面模板，不复制整本正文供人工维护。

生成脚本应先检查所有输入，再覆盖自动生成文件；删除过时输出时，只能操作明确的生成目录。不得清理 `docs/`、`config/` 或其他现有目录。

## 5. 信息架构和阅读体验

首页提供书名、简要说明、开始阅读、完整目录、原文来源、GitHub 仓库及语言切换入口。默认中文；英文入口清晰可见。页面不冒充原作者官方网站。

导航设计：

- 顶部：阅读目录、来源与说明、原文、GitHub、语言选择、主题切换。
- 左侧：前言、引言、七个 Part 下的章节、参考文献；依据 `chapters.json` 顺序和 `parent` 生成。
- 右侧：当前页二、三级标题目录；较深层标题留在正文中。
- 正文：来源提示、标题、内容、上一条/下一条、编辑原始 Markdown 链接。
- 移动端：可折叠菜单；长公式和宽表格允许横向滚动；图片限制最大宽度。

使用 VitePress 默认主题并做少量阅读样式调整。正文宽度、中文行距及代码字体以长文阅读为先，不添加无关首页动画或后台服务。

前置内容中可能只有二级书名标题，部分章节则包含 `Chapter N` 和一级标题。生成阶段统一页面主标题规则，保证每页只有一个主要 H1；不得因此删除正文段落或破坏原锚点。

## 6. 中英文路由和切换

### 6.1 URL 约定

中文使用站点根目录，英文使用 `/en/`。这样无需依赖 GitHub Pages 不提供的自定义服务器重定向。VitePress 的 `locales.root` 和 `locales.en` 分别配置语言、标题及主题文案。[国际化配置参考](https://vuejs.github.io/vitepress/v1/guide/i18n)

以下均为去掉 Pages 仓库前缀后的逻辑路径：

| 内容 | 中文 | 英文 |
| --- | --- | --- |
| 首页 | `/` | `/en/` |
| 引言 | `/01-introduction/` | `/en/01-introduction/` |
| Part I | `/part-i-language-modeling/` | `/en/part-i-language-modeling/` |
| 参考文献 | `/90-references/` | `/en/90-references/` |
| 来源说明 | `/source/` | `/en/source/` |

章节目录下使用 `index.md`，对应真实 `index.html`，公开链接统一以 `/` 结尾；保留 `cleanUrls: false`，避免依赖服务器为扩展名缺失的页面做改写。

### 6.2 同章切换行为

1. 生成 `内容 ID → 中文路由、英文路由、原始文件` 的显式映射。
2. 使用该映射驱动唯一的顶部语言切换控件，替换默认主题对应入口，避免出现两个不同规则的语言菜单。
3. 在中文引言切换 English，进入英文引言；切换回来仍在中文引言，不跳回首页。
4. 目标页存在同名 `page-*` 锚点时可保留该片段；若标题锚点已建立双语映射，转换为目标锚点；否则定位目标章节顶部。
5. 同名锚点存在只保证链接可用，不代表两种语言精确对应到同一段落。首版承诺同章切换，不承诺逐句对齐或滚动位置同步。
6. 构建时发现任一章节缺少译文即失败；独立网站页面若缺少配对，可回退目标语言首页并明确标示。
7. 自定义组件链接统一通过 VitePress 的 base 处理，不直接硬编码域名或仓库前缀。

中英文还需分别配置搜索提示、目录标题、前后页按钮、移动导航及来源说明，并检查最终 HTML 的 `lang` 属性。

## 7. 内容转换与资源处理

### 7.1 内容白名单

正文只读取 `config/chapters.json` 指定的 32 组文件。首页参考现有索引生成目录。`docs/marker-pdf-to-bilingual-markdown-plan.md`、根目录方案、校验报告和工作记录不得自动成为网站页面或进入搜索索引。

### 7.2 链接与锚点

- 解析 Markdown 链接、引用式链接和 HTML 中的有效链接，使用源路径解析目标，再查映射改写为网站路由。
- 例如 `../01-introduction/01-introduction-en.md#11-why-this-book` 应变为 `/en/01-introduction/#11-why-this-book`，并实际验证输出 HTML 中的目标 ID。
- 保留原始 `page-*` 显式 ID 和文献引用。移除展示用分页分隔线时不能连同锚点一起删除。
- VitePress 自动标题 ID 可能与已有校验脚本不同；为旧链接保留兼容锚点或建立确定的重映射，不能只依靠原 Markdown 校验报告。
- 最终检查页面 ID 唯一性、站内文件存在性及所有片段链接。VitePress 的死链检查保持开启，补充构建后锚点检查。
- 外链原样保留；临时网络错误记录重试，不把论文链接失联误判为正文生成失败。

### 7.3 图片

现有 `docs/<id>/assets/<name>.jpeg` 复制到 `site/content/public/media/<id>/<name>.jpeg`。中英文页面引用同一资源，资源 ID 包含章节，避免不同章中 `_page_0_*` 等文件名冲突。

Markdown 图片路径统一改写为 `/media/<id>/<name>.jpeg` 并交由 VitePress 处理 base；原始 HTML 图片在生成阶段转换或显式处理 base。验收时从实际 `/仓库名/` 子路径加载页面，检查请求地址。

为图片添加可由图注确定的替代文本；装饰性图片可保留空替代文本。不凭空编写图表含义。初版保留原 JPEG，启用懒加载；如后续压缩，需核验图中文字可读性。

### 7.4 数学、代码和转换噪声

公式启用 `markdown.math: true` 并安装 `markdown-it-mathjax3`，这是 1.x 文档列出的组合。[数学公式支持](https://vuejs.github.io/vitepress/v1/guide/markdown#math-equations)

实施时检查行内/块级公式、上下标、矩阵、编号及超长公式；不通过全局正则替换所有 `$` 或反斜杠。代码块不得被当作正文清洗。

仅移除确定的 PDF 分页分隔行，保留页码追踪元数据。检查非代码区域中的 `<`、`{{ }}` 等是否被 Vue 当作模板解析，并只对受影响文本做转义。允许的 `<span id>` 必须保留；其他原始 HTML 需经过渲染抽查。

### 7.5 可追溯性

每页生成元数据：内容 ID、语言、源文件路径、PDF 物理页范围、来源 URL、源文件内容摘要及翻译状态。生成内容带“自动生成，请编辑 docs 源文件”注释。

编辑链接直接指向 `docs/<id>/<id>-zh.md` 或 `-en.md`。首版不显示生成文件的虚假更新时间；若后续增加，取原始源文件的 Git 最近修改时间。

## 8. 来源标注、引用和翻译说明

### 8.1 来源元数据

根据本次核查，原文标题为 **Agentic Large Language Models**，作者为 **Max van Duijn、Michiel van der Meer、Aske Plaat、Niki van Stein**。来源页显示 v1；本地 PDF 是否对应该版本应在实施前核对，不能仅凭文件名认定。[原文详情](https://arxiv.org/abs/2610.13455)

来源页当前的许可证链接指向 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)。实施时记录该链接及核查日期，并核对本地 PDF 的版权页和单独署名材料。站点代码的许可、原文来源声明、翻译说明分别记录，不将第三方资料笼统标为项目原创。

### 8.2 必须展示的位置

| 位置 | 必须展示的内容 |
| --- | --- |
| 两种语言首页 | 原书名、作者、arXiv 来源链接、非官方整理说明 |
| 每个正文页顶部 | 可见来源链接、当前语言、译文状态或英文提取说明 |
| 来源说明页 | 完整引用、转换过程、版本核验信息、翻译状态、纠错方式 |
| 页脚 | 简短来源链接及来源说明入口 |
| GitHub README / README.en.md | 项目目标、网站地址、来源、翻译状态、本地运行方法 |
| THIRD_PARTY_NOTICES.md | 原文及第三方材料的归属、已核查许可信息 |

VitePress 默认页脚不一定在带侧栏的阅读页显示，因此来源标注以正文顶部组件为必需保障，不能只放在全局 footer。

建议中文文案：

> 内容来源：[Agentic Large Language Models（arXiv:2610.13455）](https://arxiv.org/abs/2610.13455)。本页为非官方中文翻译与网页整理，译文待人工复核；内容理解请对照英文原文。转换与翻译可能存在误差。

建议英文文案：

> Source: [Agentic Large Language Models (arXiv:2610.13455)](https://arxiv.org/abs/2610.13455). This is an unofficial web edition extracted from the source PDF. Formatting and extraction errors may remain; please consult the original.

所有来源文案使用同一元数据配置，避免 64 页各自维护。出处必须是可见、可点击的链接，不能只藏在 HTML 注释中。

## 9. 搜索与基础 SEO

使用 VitePress 本地搜索，索引正文标题与内容，优先展示当前语言结果；搜索界面文案按语言配置。[本地搜索配置](https://vuejs.github.io/vitepress/v1/reference/default-theme-search)

中文搜索需专门验收“智能体”“强化学习”“工具调用”等词，英文验收 “agent”“scaling laws”“retrieval”。若默认分词不能满足中文词组召回，通过 MiniSearch 的分词配置加入中英文兼容处理，例如中文二元切分加英文词切分，并使建索引与查询使用同一规则。不要把搜索框出现等同于中文搜索可用。

每页提供语言对应的标题和摘要。最终确定域名与仓库路径后，生成 canonical、双语 alternate/hreflang 和 sitemap；URL 必须含正确 Pages base。404 页面提供中英文首页和目录入口。

站点以静态内容为主，不引入登录、评论、数据库或访问统计。搜索索引、图片、字体均随站点提供，避免阅读依赖额外服务。

## 10. pnpm 命令和配置约定

### 10.1 脚本接口

以下是拟定义的命令接口，不代表脚本已经存在：

| 命令 | 约定行为 |
| --- | --- |
| `pnpm install` | 首次安装并生成锁文件；仅在实施阶段执行 |
| `pnpm install --frozen-lockfile` | CI 按锁文件安装，版本漂移时失败 |
| `pnpm docs:generate` | 从正文和页面模板生成完整网站内容及映射 |
| `pnpm docs:check` | 检查配对关系、源文件、图片及基础链接 |
| `pnpm docs:dev` | 先生成，再启动内容监听器和 VitePress 开发服务 |
| `pnpm docs:build` | 输入校验 → 生成 → `vitepress build site` |
| `pnpm docs:check-dist` | 检查生成 HTML、图片、锚点和路由 |
| `pnpm docs:preview` | 执行 `vitepress preview site`，预览已构建产物 |

监听器只监听内容源、配置及页面模板，不监听自己的生成目录，避免无限循环。开发命令停止时同时结束监听器与开发服务。

### 10.2 配置骨架

此代码仅说明配置位置与关键选择，菜单、来源组件及切换逻辑仍按前文设计实现。

```ts
import { defineConfig } from 'vitepress'

export default defineConfig({
  srcDir: 'content',
  // 示例值：实施时由真实仓库名确定
  base: '/AgenticLargeLanguageModels/',
  cleanUrls: false,
  ignoreDeadLinks: false,
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: '智能体大语言模型',
      description: 'Agentic Large Language Models 非官方中文翻译与阅读站'
    },
    en: {
      label: 'English',
      lang: 'en',
      title: 'Agentic Large Language Models',
      description: 'An unofficial bilingual reading edition'
    }
  },
  markdown: {
    math: true,
    image: { lazyLoading: true }
  },
  themeConfig: {
    search: { provider: 'local' },
    outline: [2, 3]
  }
})
```

依赖安装、构建和预览统一使用 pnpm；不同时生成 `package-lock.json` 或 `yarn.lock`。不在配置里写访问令牌。`owner`、`repo`、生产站点地址统一存于站点元数据，构建时拒绝遗留占位符。

## 11. GitHub 仓库与 Pages 发布

### 11.1 仓库约定

建议仓库名 `AgenticLargeLanguageModels`，默认分支 `main`，初版使用公开仓库和 GitHub 项目 Pages。最终 owner、仓库名与可见性待实施阶段确定，不从本机用户名推断 GitHub 账号。

常规项目站地址：`https://<OWNER>.github.io/<REPO>/`，对应 `base: '/<REPO>/'`。若采用 `<OWNER>.github.io` 用户站仓库，base 为 `/`。名称大小写及路径必须与实际仓库一致。[部署路径说明](https://vitepress.dev/guide/deploy#setting-a-public-base-path)

首次提交应先设置忽略规则，并审查待提交清单：

```gitignore
node_modules/
.pnpm-store/
.venv/
__pycache__/
*.pyc
work/
tmp/
site/content/
site/.vitepress/generated/
site/.vitepress/cache/
site/.vitepress/dist/
.env
.env.*
!.env.example
AgenticLargeLanguageModels.pdf
```

保留正文、图片、章节清单、必要脚本、网站配置、锁文件、README 和本方案。`reports/` 仅提交审核过且有维护价值的报告；本地模型、翻译缓存、虚拟环境及临时提取文件不进入发布仓库。原 PDF 默认通过 arXiv 链接访问。

### 11.2 CI 与部署分离

| 工作流 | 触发 | 权限 | 内容 |
| --- | --- | --- | --- |
| `ci.yml` | PR、推送 main | `contents: read` | 冻结安装、输入校验、生成、构建、输出校验；不部署 |
| `deploy.yml` | 推送 main、手动运行 | 构建只读；部署增加 Pages/OIDC 权限 | 同样验证产物，随后上传和发布 |

Pages 使用仓库 Settings → Pages → Build and deployment → Source → **GitHub Actions**。不维护 `gh-pages` 产物分支，也不把生成的 HTML 手工提交到 `docs/`。[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

部署工作流顺序：

1. 检出源码；将 Node 配置为选定版本。
2. 使用 `pnpm/action-setup` 按 `packageManager` 安装固定 pnpm；启用缓存前确保 pnpm 已可运行。
3. 配置 pnpm store 缓存，缓存键含系统、pnpm 版本及锁文件摘要；初版不缓存 VitePress 生成目录。
4. `pnpm install --frozen-lockfile`。
5. `pnpm docs:build`，失败即停止。
6. `pnpm docs:check-dist`，失败即停止。
7. 使用 `actions/configure-pages` 配置部署元数据。
8. 使用 `actions/upload-pages-artifact` 上传 **`site/.vitepress/dist/`**，不得上传整个仓库。
9. 独立 deploy job 依赖成功的 build job，使用 `actions/deploy-pages` 发布。
10. deploy job 使用 `github-pages` environment，将输出的 `page_url` 记录为部署地址。

部署 job 授予 `pages: write` 和 `id-token: write`；使用 GitHub 提供的令牌，不要求额外个人访问令牌。工作流 Action 版本按实施时官方文档核验并固定；部署并发组为 `pages`，`cancel-in-progress: false`，防止打断已开始的发布。

手动部署也必须限制在 `main`，例如 job 条件校验 `github.ref == 'refs/heads/main'`。PR 工作流不能触发生产发布；不使用具有写权限的 PR 工作流执行来自外部贡献者的代码。

### 11.3 首次发布顺序

1. 完成所有本地实现和验收，形成可预览的网站及明确的提交清单。
2. 确定真实 GitHub owner、repo 和公开范围，写入站点元数据。
3. 初始化 Git、建立 `main`，审查后提交源码；创建或连接对应远程仓库。
4. 推送源码，配置 Pages 来源和环境；如首次推送先于 Pages 设置完成，则设置后手动重跑部署。
5. 查看构建、产物上传和部署结果，记录 workflow run URL 和站点 URL。
6. 在线检查首页、深层路由刷新、语言切换、图片、公式、搜索及出处链接。
7. 将真实站点地址补入双语 README 和仓库 About，交付仓库及网站链接。

公开发布属于后续实施阶段；本方案不执行这些操作。

## 12. 验收标准

| 范围 | 必须通过的验收 |
| --- | --- |
| 内容数量 | 32 组条目全部生成，64 个正文页完整配对；另有双语首页和双语来源页 |
| 来源标注 | 全部正文页、双语首页、来源页及 README 均可点击到指定 arXiv 页面 |
| 翻译说明 | 中文页面明确为非官方译文；尚未完成人工复核时如实显示 |
| 导航 | 按清单顺序分组，Part 可阅读；上一条/下一条不跨语言 |
| 语言切换 | 全部 32 组页面双向切换到同一内容 ID；移动端同样可用 |
| 图片 | 120 张来源图片复制成功，无同名覆盖；所有实际引用均能加载 |
| 公式与代码 | 训练、规模定律、推理等代表页面抽检通过，无公式源码异常或代码损坏 |
| 内部链接 | 自动遍历产物中的站内链接、资源和片段，无缺失文件和无效 ID |
| HTML 锚点 | 无重复 ID；旧目录和参考文献链接仍指向正确位置 |
| 深层访问 | 从地址栏直接访问或刷新章节 URL 仍成功，不依赖先访问首页 |
| Pages base | 使用真实仓库前缀验证 CSS、JS、搜索索引、图片和语言切换 |
| 搜索 | 至少 5 个中文和 5 个英文代表词可找到预期页面；中文词组可召回 |
| 响应式 | 约 375px 手机宽度和桌面宽度均可阅读；表格与公式不撑坏页面 |
| 基础可访问性 | 键盘能操作导航与语言切换，焦点可见，图片替代文本合理 |
| 发布范围 | 产物及搜索索引不含工作日志、方案文档、模型和缓存 |
| 可复现性 | 全新检出可通过冻结安装完成构建；生成结果不依赖本机绝对路径 |
| 在线发布 | 主分支工作流成功，Pages HTTPS URL 可打开，部署记录可追溯到提交 |

自动检查负责所有页面链接和资源；人工检查负责排版、公式、翻译说明及搜索质量。源码和构建产物检查已通过，部分桌面浏览器交互也已抽查；在线发布及线上验收尚未执行。

## 13. 实施阶段与交付物

| 阶段 | 工作内容 | 完成标志 |
| --- | --- | --- |
| 1. 基础核对 | 固定版本、记录来源、编写忽略规则 | 已完成 |
| 2. 网站工程 | 建立 pnpm/VitePress 工程并生成双语页面 | 已完成 |
| 3. 全量生成 | 图片复制、链接改写和菜单生成 | 已完成，32 组内容全部出现 |
| 4. 双语体验 | 同章切换、搜索、来源组件和样式 | 已完成；已抽查页面切换、公式与图片 |
| 5. 产物校验 | 链接、锚点、子路径及资源验证 | 已完成，构建与产物校验通过 |
| 6. GitHub 发布 | 源码提交、远程仓库、Actions 和 Pages | 待确定 GitHub owner/repo 并发布 |
| 7. 交付维护 | 双语 README、发布记录、更新指南 | README 和来源说明已完成；线上记录待发布后补充 |

最终交付：GitHub 源码仓库、GitHub Pages 网站、固定版本的 pnpm 工程、自动生成与校验脚本、自动部署工作流、双语使用说明，以及全部页面的来源标注。

## 14. 更新、回滚与已知风险

日常更新只编辑 `docs/` 中对应语言正文、章节图片或站点页面模板。章节结构变化同步更新 `config/chapters.json`。PR 检查通过后合并 main，部署工作流重新生成全站。

发布失败时保留上次成功部署，先根据失败日志修复。已发布内容有误时，通过回退对应源提交产生新提交，再由工作流重新构建发布；不手工篡改构建产物。若只是平台临时失败，可重跑对应工作流。

| 风险 | 处理方式 |
| --- | --- |
| 翻译仍待人工复核 | 保留状态提示和原文入口，逐章校对后再更新状态 |
| VitePress 与既有标题锚点规则不同 | 构建后检查实际 HTML ID，必要时加入兼容锚点 |
| PDF 提取残留影响 Vue 编译 | 先构建代表页面，针对真实失败点处理，不做广泛文本替换 |
| 中文搜索召回不足 | 单独验证中文分词与索引，确保实现和查询规则一致 |
| base 设置错误 | 在仓库子路径进行本地和线上验收，禁止硬编码根域路径 |
| 误提交模型或临时文件 | 初始化 Git 前写忽略规则，首次提交检查文件清单及体积 |
| 生成目录成为第二份内容源 | 忽略生成目录，编辑入口指向原始 Markdown |
| GitHub Pages 体积或构建时间超限 | 构建时记录产物体积和耗时，对照[官方限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)检查 |

## 15. 后续需要确定的参数

本地实施已完成；连接 GitHub 远程前仍需落实：

- GitHub 账号或组织：`<OWNER>`。
- 仓库名：建议 `AgenticLargeLanguageModels`，以实际创建结果为准。
- 可见性：建议公开；如选择私有，核验账号计划下的 Pages 可用性及网站公开范围。
- 站名：建议中文“智能体大语言模型”，英文“Agentic Large Language Models”。
- 语言策略：本方案默认中文在根路径、英文在 `/en/`。
- 域名：首版使用 GitHub Pages 默认域名；自定义域名可后续配置并重建 URL 元数据。
- 原文版本：核对本地 PDF 与 arXiv 版本，记录文件摘要和来源版本。

## 16. 参考资料

以下资料于 2026-10-16 查阅；实施时再次核验依赖和 Action 版本。

1. [原始资料：Agentic Large Language Models](https://arxiv.org/abs/2610.13455)
2. [原文页面链接的 CC0 1.0 声明](https://creativecommons.org/publicdomain/zero/1.0/)
3. [VitePress 1.x 国际化](https://vuejs.github.io/vitepress/v1/guide/i18n)
4. [VitePress 1.x Markdown 与公式](https://vuejs.github.io/vitepress/v1/guide/markdown)
5. [VitePress 1.x 本地搜索](https://vuejs.github.io/vitepress/v1/reference/default-theme-search)
6. [VitePress 部署说明](https://vitepress.dev/guide/deploy)
7. [pnpm 安装及兼容性](https://pnpm.io/installation)
8. [GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
9. [GitHub Pages 限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
