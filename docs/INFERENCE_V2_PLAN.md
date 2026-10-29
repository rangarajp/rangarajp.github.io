# Inference Engineering V2 — architecture plan

Phase A audit of this repository. **No app source, dependencies, or deployment config were changed.** Implementation waits on explicit approval.

Editorial promise: Understand → Derive → Implement → Measure → Optimize → Operate.

Existing public page to preserve: `/concepts/llm-inference/`.

New handbook base path: `/concepts/llm-inference-v2/`.

---

## 1. Current stack

| Item | Reality |
|------|---------|
| Astro | `^7.1.4` (`astro.config.mjs`, `package.json`) |
| Node | `>=22.12.0` |
| Package manager | npm (`package-lock.json`) |
| Scripts | `dev`, `build`, `preview`, `astro` only — no lint, typecheck, or test scripts |
| Integrations | `@astrojs/mdx` `^7.0.4`, `@astrojs/sitemap` `^3.7.3` |
| Markdown | `remark-math` + `rehype-katex` via `@astrojs/markdown-remark` unified processor |
| Images | `sharp` |
| RSS | `@astrojs/rss` present, unused by inference routes |
| Starlight | **not installed** |
| Pagefind | **not installed** |
| Site URL | `https://rangarajp.github.io` (user site, **no `base` path**) |
| `trailingSlash` | unset (Astro default: ignore both with/without slash) |
| Fonts | local Atkinson via `fontProviders.local()` |

There is no TypeScript path alias beyond defaults, no React/Vue/Svelte, and no global client framework.

---

## 2. Folder tree that matters

```text
src/
  content.config.ts                 # single collection: concepts
  content/concepts/
    llm-inference/                  # existing book (must stay)
    llm-system-design/
    transformers/
    reasoning-models/
  pages/concepts/[...slug].astro    # all concept URLs
  pages/concepts/index.astro        # /concepts book grid
  lib/concepts.ts                   # series grouping, paths
  components/                       # Header, sidebar, TOC, section search
  layouts/                          # none (pages own markup)
  styles/global.css, prose.css
notebooks/llm-inference/            # labs, serving, experiments (not routed)
.github/workflows/deploy.yml        # Pages: withastro/action@v5 on main
```

Content collections: **one** collection named `concepts`, glob `./src/content/concepts/**/*.{md,mdx}`. Schema is thin: `title`, `description`, `pubDate`, optional `updatedDate`, `heroImage`, `order`, `seriesOrder`. No part, difficulty, status, or tags.

---

## 3. How `/concepts/llm-inference/` is generated

1. Files under `src/content/concepts/llm-inference/*.md` enter the `concepts` collection.
2. Hub `index.md` (`seriesOrder: 2`) becomes `/concepts/llm-inference` via `getConceptPath`.
3. Chapters with `order` become `/concepts/llm-inference/<filename>`.
4. `src/pages/concepts/[...slug].astro` renders every collection entry with `Header`, `ConceptsBookSidebar` (all books), article `prose`, in-page `SectionSearch` (heading jump), `PageToc`, prev/next.

Assets: chapter-local `images/` plus `Inference_Math_Basics.xlsx`. Hero images point at `src/assets/blog-placeholder-3.jpg`.

Hub currently lists nine ordered chapters (plus `order: 1.5` for mathematics). `llm_inference_engineering_concepts_summary_fixed.md` lives in the same folder **without YAML frontmatter**; it is inside the collection glob and is a build risk if the loader does not skip it.

---

## 4. Navigation, search, layout, math, code

| Feature | Current behavior | Gap vs V2 spec |
|---------|------------------|----------------|
| Site header | Home, Concepts, Build from Scratch, Projects, Explore | No handbook-scoped nav |
| Book sidebar | Left, all concept series, mobile drawer | Not part-grouped; not isolated |
| In-page TOC | Right rail, h2/h3, desktop only | No Concept/Math/Code/Measure jump chips |
| Search | `SectionSearch`: **current-page headings only**, `/` shortcut | Not full-text; not handbook-scoped; no Ctrl/Cmd+K |
| Breadcrumbs | Concepts / series / title | Exists |
| Prev/next | Chapter order | Exists |
| Math | KaTeX globally via markdown config | Keep; do not duplicate processor |
| Code | Astro default highlighting | **No copy button** |
| Diagrams | Markdown images / HTML figures | No extra diagram lib |
| Dark mode | Light-only (`global.css`) | Do not invent a second theme in V1 |
| Copy / syntax | Shiki via Astro | Copy is V1 add, scoped to v2 layout |

---

## 5. GitHub Pages, base URL, trailing slash

- Workflow: push/`workflow_dispatch` on **`main`** → `withastro/action@v5` → `actions/deploy-pages@v4`.
- `site: 'https://rangarajp.github.io'`; no `base`. Paths like `/concepts/...` are correct.
- Trailing slash: leave unset so new URLs match old ones.
- **Do not merge or deploy from this phase.** Feature work should land on `feature/inference-v2` after approval.

---

## 6. Starlight vs extend current Astro

**Recommendation: extend the current Astro site. Do not add Starlight.**

Reasons:

- Starlight is a full docs theme (its own sidebar, search, CSS, content collections). It would fight `Header`, `prose.css`, Atkinson, and the concepts handbook shell.
- The spec forbids a new public root and requires GitHub Pages compatibility; Starlight typically wants `/docs` or a takeover of `src/content`.
- V1 needs isolation, not a second design system. Existing pieces already cover sidebar, TOC, breadcrumbs, KaTeX, MDX, and static Pages deploys.
- Adding Starlight is a large shared-config change; the spec asks for the smallest compatible set.

Revisit Starlight only if the handbook later outgrows a custom collection + layout.

---

## 7. Proposed files (after approval — Phase B)

Isolated collection so v2 **does not** appear inside the old book sidebar (that sidebar lists every `concepts` series).

```text
src/content.config.ts                          # ADD collection inferenceV2 only
src/content/inference-v2/
  index.md                                     # landing
  learn/prefill-vs-decode.md                   # sample chapter (V1)
  learn/_planned/*.md                          # optional stubs with status: planned
  reference/index.md
  journal/index.md
src/lib/inference-v2.ts                        # nav tree, paths, search index helper
src/pages/concepts/llm-inference-v2/
  index.astro
  [...slug].astro
src/components/inference-v2/
  Layout.astro
  Sidebar.astro
  PageToc.astro
  SearchDialog.astro
  ChapterMeta.astro
src/styles/inference-v2.css                    # scoped; no global.css edits
docs/INFERENCE_V2_PLAN.md                      # this file (already)
docs/INFERENCE_V2_CONTENT_MAP.md               # already
```

**Unavoidable shared config (minimal):**

1. `src/content.config.ts` — register `inferenceV2` with a richer schema (`part`, `chapter`, `difficulty`, `status`, `tags`, `prerequisites`, `related`).
2. Optional later (needs approval): a single “See also” link on the *old* hub or a card on `/concepts`. **Not required for V1.** Do not change `Header.astro` in V1 unless approved.

**Do not:**

- Put files under `src/content/concepts/llm-inference-v2/` (they would join the existing collection and leak into every concepts sidebar).
- Redirect `/concepts/llm-inference/` .
- Edit `src/styles/global.css` or `prose.css`.
- Change `deploy.yml` or `astro.config.mjs` markdown/fonts unless KaTeX/MDX already covers v2 (they will, if the new pages use the same markdown pipeline via content collections).

`experiments/inference-v2/` is Phase C+, not V1.

### Search (V1)

Build a static JSON index from `inferenceV2` bodies + headings at build time; client dialog scoped to `/concepts/llm-inference-v2/**`. Support Ctrl/Cmd+K on those pages.

Prefer this over Pagefind for V1: no extra npm binary, no second indexer, still static. Add Pagefind later if the corpus grows large.

### Isolation

- Routes only under `/concepts/llm-inference-v2/`.
- CSS imported from v2 layout only (`inference-v2.css` + existing `prose.css` via `BaseHead` is acceptable for typography consistency; do not restyle `.prose` globally).
- Components under `src/components/inference-v2/`.
- Reuse `BaseHead` (KaTeX CSS, fonts, sitemap) — that is shared by design and does not alter old pages.

### Risks and rollback

| Risk | Mitigation |
|------|------------|
| New collection breaks `getCollection('concepts')` | Keep collections separate; do not change `lib/concepts.ts` |
| Sidebar leak | Never glob v2 under `src/content/concepts/` |
| Global CSS regression | No edits to `global.css` / `prose.css` |
| Search bloat | Index v2 only; small client script on v2 layout |
| Pages deploy | Branch work only; `main` unchanged until you merge |
| Frontmatter-less md in old folder | Leave as-is in V1; do not copy that pattern |

Rollback: delete the v2 tree + revert `content.config.ts`. Old routes untouched.

---

## 8. Route and IA (V1)

| URL | Role | V1 content |
|-----|------|------------|
| `/concepts/llm-inference-v2/` | Home: promise, six-part map, tracks, search | Real landing |
| `/concepts/llm-inference-v2/learn/prefill-vs-decode/` | Sample canonical chapter | Substantive (rewrite, not a copy of the old page) |
| `/concepts/llm-inference-v2/learn/` | Optional learn index | Part list + honest statuses |
| `/concepts/llm-inference-v2/reference/` | Glossary/formulas placeholder | Empty state, not fake articles |
| `/concepts/llm-inference-v2/journal/` | Build journal index | Empty state |

Planned chapters appear in the sidebar as *planned*, not as padded stubs. Flagship chapters (inference mathematics, KV cache) wait for Phase C.

Chapter template (sections may be marked planned):

1. What problem are we solving?
2. Intuition
3. Under the hood
4. Mathematics
5. Build it
6. Optimize it
7. Measure it
8. Engineering trade-offs
9. Check understanding
10. Related concepts and sources

On-page TOC plus jump links to Concept / Math / Code / Measure. Prefer readable article flow over tabs.

---

## 9. V1 checklist and acceptance tests

Work on branch `feature/inference-v2`. Do not merge or deploy.

### Build

- [ ] `npm run build` succeeds
- [ ] `dist/concepts/llm-inference/` still present and unchanged in content
- [ ] `dist/concepts/llm-inference-v2/` contains landing + sample chapter

### Routes

- [ ] `/concepts/llm-inference/` and existing chapter URLs still 200
- [ ] `/concepts/llm-inference-v2/` 200
- [ ] `/concepts/llm-inference-v2/learn/prefill-vs-decode/` 200
- [ ] No redirect from old → new
- [ ] Trailing-slash behavior matches the rest of the site

### UX

- [ ] Left sidebar: Home / Learn (parts) / Reference / Journal; mobile drawer accessible
- [ ] Right TOC on wide screens; headings have anchors
- [ ] Search finds a term **inside sample chapter body** (built site, not only `astro dev`)
- [ ] Ctrl/Cmd+K opens handbook search on v2 pages
- [ ] Keyboard: skip, focus rings, Esc closes search/drawer
- [ ] Planned chapters show empty/planned state, not fake “complete” copy

### Isolation

- [ ] `git diff` on `src/content/concepts/llm-inference/` is empty
- [ ] `global.css` / `prose.css` / `Header.astro` / `deploy.yml` unchanged (unless a listed exception was approved)
- [ ] Old concepts sidebar does not list v2 chapters

### Content quality

- [ ] No fabricated benchmarks or hardware claims
- [ ] Illustrative math labeled as such
- [ ] O'Reilly markdown rules for new chapter prose

---

## 10. Phase sequence (do not skip)

| Phase | Scope | Status |
|-------|--------|--------|
| A | Audit + these docs | **This delivery** — stop for approval |
| B | Isolated shell, search, one sample chapter | After you approve this plan |
| C | Prefill vs decode polish, inference mathematics, KV cache | After B acceptance |
| D | Calculators / visualizations | After C |
| E | Serving and operations | After D |

Hands-on code under `notebooks/llm-inference/` is real (vLLM, FastAPI gateway, multi-model serving, KV experiments). Phase C/E may link to it **after** verifying versions and results. Do not paste unverified numbers into the handbook.

---

## 11. Decision points needing your approval

1. **No Starlight** — extend current Astro (recommended).
2. **Separate `inferenceV2` collection** — not a subfolder of `concepts`.
3. **No site-header link in V1** — handbook reachable by URL; optional `/concepts` card later.
4. **Custom static search JSON** instead of Pagefind for V1.
5. **Create `feature/inference-v2` and implement Phase B** after you say go.

If any of these is wrong, say so before implementation.
