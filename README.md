# jaisharmz.github.io

Personal academic site for Jai Sharma. Live at https://jaisharmz.github.io

Static HTML/CSS, no external assets. The home page follows the Jon Barron academic
template; the writing pages follow Lil'Log (one long post) and the JAX Scaling Book (a book
as chapters with a sidebar).

```
index.html            home page, hand written
writing/              generated: posts and the book, from src/ by tools/build.mjs
assets/css/style.css  design system (light + dark, follows system or toggle)
assets/css/writing.css  posts, book chapters, interactive figures
assets/js/main.js     theme toggle
assets/js/writing.js  contents highlight while reading
assets/js/figlib.js   runtime for interactive figures
assets/js/figures/    one script per interactive figure
assets/katex/         KaTeX stylesheet and fonts, vendored by the build
assets/pdf/           hosted papers, books, white papers, CV (addresses never change)
assets/img/thumbs/    publication teasers (rendered from PDF page 1)
assets/img/writing/   generated: TikZ drawings as SVG, notebook plots, book images
src/                  LaTeX sources of the writing, writing.json, figure fragments, notebooks
tools/                the build, the figure contract (FIGURES.md), figure tests
```

## Writing pages

The PDFs stay where they are, because links to them have been sent out. The web editions
live beside them under `writing/`.

```sh
cd tools && npm install     # once: KaTeX and cheerio
node build.mjs              # rebuilds writing/ and assets/img/writing/
```

The build needs `pandoc` 3 and a TeX installation with `latex` and `dvisvgm` (for TikZ).
It renders every formula with KaTeX at build time, so pages carry no math JavaScript.

**Update a post from Overleaf.** Replace the body of `src/<slug>.tex` with the new source,
keep the `\webfig{...}` lines where figures go, and rebuild. Strip comments first, since the
repository is public.

**Add a post.** Put `src/<slug>.tex` in place, add an entry to `src/writing.json`
(`type: "post"`, title, date, dek, pdf), rebuild, and link it from the Writing section of
`index.html`.

**Book chapters.** A book entry in `writing.json` lists its chapters, each naming the
`\section` titles of the source it is made from. One section becomes a chapter whose
subsections are its sections; several sections become one chapter with a section each.
Notebook outputs replace the screenshots under code listings in chapters that name a
`notebook`.

**Interactive figures.** See `tools/FIGURES.md`. A `\webfig{id}` line in the LaTeX places
`src/figures/<id>.html` and loads `assets/js/figures/<id>.js`. Every number in a caption
comes from `node tools/figtest/<id>.mjs`.

**Preview.** `python3 -m http.server 7792` in the repository, then open
http://127.0.0.1:7792/writing/. A single figure: `/tools/figpreview.html?fig=<id>`.

## Home page

**Add a publication.** Copy an `<li class="pub">` block in `#research`. Generate a
teaser with:

```sh
pdftoppm -jpeg -r 72 -f 1 -l 1 -scale-to-x 340 -scale-to-y -1 \
  assets/pdf/NAME.pdf assets/img/thumbs/NAME
```

**Add a news item.** Prepend an `<li>` to `.news` in `#news`.

Deployed from `main` via GitHub Pages.
