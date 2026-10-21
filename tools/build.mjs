// Build the writing pages from LaTeX.
//
//   cd tools && npm install && node build.mjs
//
// src/writing.json lists the documents. Each .tex in src/ goes through:
//   1. a LaTeX pass here: drop print-only commands, compile every tikzpicture to SVG,
//      turn \webfig{id}, \includegraphics and \\ paragraph breaks into tokens pandoc keeps
//   2. pandoc, LaTeX to an HTML fragment with math left as TeX
//   3. an HTML pass here: KaTeX renders every formula at build time, tokens become
//      figures, headings get numbers and a contents list, notebook outputs go under
//      the code that produced them
//   4. a template: posts are one page, a book is a home page plus one page per chapter.
// The output is plain HTML committed next to index.html, because GitHub Pages serves
// the repository as it is. Nothing here runs in the reader's browser except figures.

import { readFileSync, writeFileSync, mkdirSync, existsSync, copyFileSync, readdirSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import katex from "katex";
import * as cheerio from "cheerio";

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(TOOLS, "..");
const SRC = path.join(REPO, "src");
const CACHE = path.join(TOOLS, ".cache");
const PANDOC = process.env.PANDOC || ["/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc"].find(existsSync) || "pandoc";
const TEXBIN = ["/Library/TeX/texbin", "/usr/local/texlive/2025/bin/universal-darwin"].find(existsSync) || "";

// typographic apostrophes and quotes for strings that come from writing.json
const sq = (s) => String(s).replace(/(\w)'(\w)/g, "$1\u2019$2").replace(/(^|\s)"/g, "$1\u201C").replace(/"/g, "\u201D");
const NO_FIGURES = process.argv.includes("--figures=none");
const config = JSON.parse(readFileSync(path.join(SRC, "writing.json"), "utf8"));
for (const d of config.docs) {
  for (const k of ["title", "dek", "venue"]) if (d[k]) d[k] = sq(d[k]);
  for (const c of d.chapters || []) { if (c.title) c.title = sq(c.title); if (c.desc) c.desc = sq(c.desc); }
}
const warnings = [];
const warn = (msg) => { warnings.push(msg); console.warn("  warning: " + msg); };

const KATEX_MACROS = {
  "\\EX": "\\operatorname{\\mathbb{E}}",
  "\\Var": "\\mathrm{Var}",
  "\\Cov": "\\mathrm{Cov}",
  "\\diff": "\\frac{\\mathrm{d}#1}{\\mathrm{d}#2}",
  "\\independent": "\\perp\\!\\!\\!\\perp"
};

// ---------------------------------------------------------------------------
// helpers

const sha = (s) => createHash("sha256").update(s).digest("hex");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const mkdir = (d) => mkdirSync(d, { recursive: true });
const write = (file, text) => { mkdir(path.dirname(file)); writeFileSync(file, text); };
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const longDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };
const year = (iso) => iso.slice(0, 4);

function renderMath(tex, display) {
  try {
    return katex.renderToString(tex, {
      displayMode: display, throwOnError: true, strict: "ignore", trust: false,
      macros: { ...KATEX_MACROS }, output: "html"
    });
  } catch (err) {
    warn(`KaTeX: ${err.message.split("\n")[0]} in ${tex.slice(0, 80)}`);
    return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", macros: { ...KATEX_MACROS }, output: "html" });
  }
}

// Render \( \) and \[ \] inside a figure fragment's title and caption.
function renderInlineDelims(html) {
  return html
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => renderMath(t, true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => renderMath(t, false));
}

// ---------------------------------------------------------------------------
// 1. LaTeX pass

function splitDoc(tex) {
  const b = tex.indexOf("\\begin{document}");
  const e = tex.lastIndexOf("\\end{document}");
  return { preamble: tex.slice(0, b), body: tex.slice(b + "\\begin{document}".length, e) };
}

// Packages a standalone TikZ picture needs, taken from the document's own preamble.
function tikzPreamble(preamble) {
  const keep = ["amsmath", "amssymb", "xcolor", "color", "tikz", "pgfplots", "neuralnetwork", "cancel", "tikzsymbols"];
  const pkgs = [...preamble.matchAll(/\\usepackage(\[[^\]]*\])?\{([^}]+)\}/g)]
    .flatMap((m) => m[2].split(",").map((p) => p.trim()))
    .filter((p) => keep.includes(p));
  const colors = [...preamble.matchAll(/\\definecolor\{[^}]+\}\{[^}]+\}\{[^}]+\}/g)].map((m) => m[0]);
  return [
    "\\def\\pgfsysdriver{pgfsys-dvisvgm.def}",
    "\\documentclass[border=3pt]{standalone}",
    "\\usepackage{amsmath,amssymb}",
    ...[...new Set(pkgs)].map((p) => `\\usepackage{${p}}`),
    ...colors
  ].join("\n");
}

function compileTikz(snippet, preamble, outDir, slug) {
  const src = `${tikzPreamble(preamble)}\n\\begin{document}\n${snippet}\n\\end{document}\n`;
  const h = sha(src).slice(0, 12);
  const name = `tikz-${h}.svg`;
  const out = path.join(outDir, name);
  const cached = path.join(CACHE, "tikz", name);
  if (!existsSync(cached)) {
    const dir = path.join(tmpdir(), `tikz-${h}`);
    mkdir(dir);
    writeFileSync(path.join(dir, "fig.tex"), src);
    const env = { ...process.env, PATH: `${TEXBIN}:${process.env.PATH}` };
    try {
      execFileSync("latex", ["-interaction=nonstopmode", "-halt-on-error", "fig.tex"], { cwd: dir, env, stdio: "pipe" });
      execFileSync("dvisvgm", ["--no-fonts", "--exact-bbox", "--precision=3", "-o", "fig.svg", "fig.dvi"], { cwd: dir, env, stdio: "pipe" });
    } catch (err) {
      warn(`TikZ in ${slug} failed to compile (${h}); see ${dir}/fig.log`);
      return null;
    }
    mkdir(path.dirname(cached));
    copyFileSync(path.join(dir, "fig.svg"), cached);
  }
  mkdir(outDir);
  copyFileSync(cached, out);
  return name;
}

// Split LaTeX into protected chunks (math, tables, code listings) and the text between them.
const PROTECTED = /(\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|tabular|lstlisting|verbatim|cases|array|bmatrix|pmatrix)\}[\s\S]*?\\end\{\2\}|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/;
function splitProtected(tex) {
  const out = [];
  const re = new RegExp(PROTECTED.source, "g");
  let last = 0, m;
  while ((m = re.exec(tex))) {
    out.push({ protected: false, text: tex.slice(last, m.index) });
    out.push({ protected: true, text: m[0] });
    last = m.index + m[0].length;
  }
  out.push({ protected: false, text: tex.slice(last) });
  return out;
}

function latexPass(body, preamble, doc) {
  const imgOut = path.join(REPO, "assets", "img", "writing", doc.slug);
  let t = body;

  // print-only commands
  t = t.replace(/\\(maketitle|tableofcontents|newpage|clearpage|noindent)\b/g, "");
  t = t.replace(/\\hspace\{100mm\}/g, "");
  t = t.replace(/\\color\{[^}]*\}/g, "");
  t = t.replace(/\\Smiley(\{\})?/g, "\u{1F642}");

  // display math that only wraps a drawing, a table or an image
  t = t.replace(/\\\[\s*(\\begin\{(tikzpicture|neuralnetwork|tabular)\}[\s\S]*?\\end\{\2\})\s*\\\]/g, "\n\n$1\n\n");
  t = t.replace(/\\\[\s*(\\includegraphics(\[[^\]]*\])?\{[^}]+\})\s*\\\]/g, "\n\n$1\n\n");

  // TikZ pictures, compiled to SVG files
  t = t.replace(/\\begin\{(tikzpicture|neuralnetwork)\}[\s\S]*?\\end\{\1\}/g, (snippet) => {
    const name = compileTikz(snippet, preamble, imgOut, doc.slug);
    if (name && !doc.coverArt && snippet.startsWith("\\begin{neuralnetwork}")) doc.coverArt = name;
    return name ? `\n\nXTIKZX ${name} XTIKZX\n\n` : "\n\n";
  });

  // images and interactive figures become tokens
  t = t.replace(/\\includegraphics(\[[^\]]*\])?\{([^}]+)\}/g, (_, __, file) => `\n\nXIMGX ${file} XIMGX\n\n`);
  t = t.replace(/\\webfig\{([^}]+)\}/g, (_, id) => `\n\nXFIGX ${id} XFIGX\n\n`);

  // the source separates paragraphs with \\ \\ ; make them real paragraphs
  t = splitProtected(t).map((p) => p.protected ? p.text : p.text
    .replace(/(^|\n)[ \t]*\\\\[ \t]*(?=\n|$)/g, "\n")
    .replace(/\\\\[ \t]*\n/g, "\n\n")
  ).join("");

  // code listings: the document sets the language once with \lstset
  const lang = (preamble.match(/\\lstset\{[\s\S]*?language=(\w+)/) || [])[1];
  if (lang) t = t.replace(/\\begin\{lstlisting\}(?!\[)/g, `\\begin{lstlisting}[language=${lang}]`);

  return t;
}

// ---------------------------------------------------------------------------
// 2. pandoc

function pandoc(tex, preamble) {
  // keep \newcommand and theorem definitions; drop packages pandoc would only ignore
  const pre = preamble.split("\n").filter((l) => /\\(newcommand|renewcommand|DeclareMathOperator|newtheorem|def\\)/.test(l)).join("\n");
  const doc = `\\documentclass{article}\n\\usepackage{amsthm}\n${pre}\n\\begin{document}\n${tex}\n\\end{document}\n`;
  return execFileSync(PANDOC, [
    "-f", "latex", "-t", "html5", "--math-method=mathjax", "--wrap=none",
    "--shift-heading-level-by=1"
  ], { input: doc, encoding: "utf8", maxBuffer: 64 << 20 });
}

// ---------------------------------------------------------------------------
// 3. HTML pass

function smartQuotes($) {
  const skip = new Set(["code", "pre", "script", "style", "kbd"]);
  const walk = (node) => {
    $(node).contents().each((_, c) => {
      if (c.type === "text") {
        c.data = c.data
          .replace(/(^|[\s(\[{—-])"/g, "$1“")
          .replace(/"/g, "”")
          .replace(/(\w)'(\w)/g, "$1’$2")
          .replace(/(^|[\s(\[{])'/g, "$1‘")
          .replace(/'/g, "’");
      } else if (c.type === "tag" && !skip.has(c.name) && !$(c).hasClass("math")) {
        walk(c);
      }
    });
  };
  walk($.root());
}

function loadNotebook(name) {
  const nb = JSON.parse(readFileSync(path.join(SRC, "notebooks", name), "utf8"));
  return nb.cells.filter((c) => c.cell_type === "code" && (c.outputs || []).length)
    .map((c) => ({ src: [].concat(c.source).join(""), outputs: c.outputs, used: false }));
}

const trigrams = (s) => {
  const n = s.replace(/\s+/g, "");
  const set = new Map();
  for (let i = 0; i < n.length - 2; i++) { const g = n.slice(i, i + 3); set.set(g, (set.get(g) || 0) + 1); }
  return set;
};
function similarity(a, b) {
  const A = trigrams(a), B = trigrams(b);
  let inter = 0, total = 0;
  for (const [g, c] of A) { inter += Math.min(c, B.get(g) || 0); total += c; }
  for (const c of B.values()) total += c;
  return total ? 2 * inter / total : 0;
}

function renderOutputs(outputs, outDir, slug) {
  const parts = [];
  for (const o of outputs) {
    if (o.output_type === "stream") {
      const text = [].concat(o.text).join("").replace(/\x1b\[[0-9;]*m/g, "");
      if (text.trim()) parts.push(`<pre>${esc(text.replace(/\n+$/, ""))}</pre>`);
    } else if (o.data) {
      if (o.data["image/png"]) {
        const b64 = [].concat(o.data["image/png"]).join("").replace(/\s/g, "");
        const name = `nb-${sha(b64).slice(0, 12)}.png`;
        write(path.join(outDir, name), Buffer.from(b64, "base64"));
        parts.push(`<img src="@ROOT@assets/img/writing/${slug}/${name}" alt="Plot produced by the code above" loading="lazy">`);
      } else if (o.data["text/html"]) {
        const $h = cheerio.load([].concat(o.data["text/html"]).join(""), null, false);
        $h("style, script").remove();
        const table = $h("table").first();
        if (table.length) {
          table.removeAttr("border").addClass("dataframe");
          parts.push(`<div class="table-wrap">${$h.html(table)}</div>`);
        }
      } else if (o.data["text/plain"]) {
        parts.push(`<pre>${esc([].concat(o.data["text/plain"]).join(""))}</pre>`);
      }
    }
  }
  return parts.length ? `<div class="nb-out">${parts.join("")}</div>` : "";
}

let figureCounter = 0;

function htmlPass(html, doc, opts) {
  opts = { starred: doc.starred, ...opts };
  const $ = cheerio.load(html, null, false);
  const imgOut = path.join(REPO, "assets", "img", "writing", doc.slug);
  const figures = [];

  smartQuotes($);

  // math
  $("span.math").each((_, el) => {
    const $el = $(el);
    const display = $el.hasClass("display");
    let tex = $el.text().trim();
    tex = tex.replace(/^\\\(|\\\)$/g, "").replace(/^\\\[|\\\]$/g, "");
    $el.replaceWith(renderMath(tex, display));
  });

  // keep "x," together: a line break between inline math and its punctuation reads as a typo
  $(".katex").each((_, k) => {
    if ($(k).parent().hasClass("katex-display")) return;
    const next = k.next;
    if (next && next.type === "text") {
      const m = next.data.match(/^[,.;:)!?\u201D\u2019]+/);
      if (m) {
        next.data = next.data.slice(m[0].length);
        $(k).wrap('<span class="nowrap"></span>');
        $(k).after(m[0]);
      }
    }
  });

  // a paragraph holding only display math loses its <p> margin to the display block
  $("p").each((_, p) => {
    const $p = $(p);
    if ($p.children().length === 1 && $p.children(".katex-display").length && !$p.text().replace($p.children().text(), "").trim()) {
      $p.replaceWith($p.html());
    }
  });

  // tokens
  const tokenRe = /^\s*X(FIG|IMG|TIKZ)X\s+(\S+)\s+X\1X\s*$/;
  $("p").each((_, p) => {
    const $p = $(p);
    const m = $p.text().match(tokenRe);
    if (!m) return;
    const [, kind, arg] = m;
    if (kind === "TIKZ") {
      const svg = readFileSync(path.join(imgOut, arg), "utf8");
      const wpt = parseFloat((svg.match(/<svg[^>]*\swidth=['"]([\d.]+)pt/) || [])[1] || "0");
      const hpt = parseFloat((svg.match(/<svg[^>]*\sheight=['"]([\d.]+)pt/) || [])[1] || "0");
      const scale = 1.333 * 1.45;
      const dims = wpt ? ` width="${Math.round(wpt * scale)}" height="${Math.round(hpt * scale)}"` : "";
      $p.replaceWith(`<figure class="tikz"><img src="@ROOT@assets/img/writing/${doc.slug}/${arg}"${dims} alt="Diagram from the original text" loading="lazy"></figure>`);
    } else if (kind === "IMG") {
      // a code screenshot after a listing: prefer the notebook's real output
      if (opts.notebook && /^code\d+\.png$/.test(arg)) {
        const code = $p.prevAll("div.sourceCode, pre").first();
        if (code.length) {
          const text = code.text();
          let best = null, score = 0;
          for (const cell of opts.notebook) {
            if (cell.used) continue;
            const s = similarity(text, cell.src);
            if (s > score) { score = s; best = cell; }
          }
          if (best && score >= 0.6) {
            const out = renderOutputs(best.outputs, imgOut, doc.slug);
            if (out) { best.used = true; $p.replaceWith(out); return; }
          }
        }
      }
      const src = path.join(SRC, "img", doc.slug, arg);
      if (!existsSync(src)) { warn(`missing image ${arg} for ${doc.slug}`); $p.remove(); return; }
      mkdir(imgOut);
      copyFileSync(src, path.join(imgOut, arg));
      const isCode = /^code\d+\.png$/.test(arg);
      $p.replaceWith(isCode
        ? `<div class="nb-out"><img src="@ROOT@assets/img/writing/${doc.slug}/${arg}" alt="Output of the code above" loading="lazy"></div>`
        : `<figure class="figure-img"><img src="@ROOT@assets/img/writing/${doc.slug}/${arg}" alt="Figure from the original text" loading="lazy"></figure>`);
    } else if (kind === "FIG") {
      // --figures=none publishes the text alone while figures are still being checked
      if (NO_FIGURES) { $p.remove(); return; }
      const frag = path.join(SRC, "figures", `${arg}.html`);
      const js = path.join(REPO, "assets", "js", "figures", `${arg}.js`);
      if (!existsSync(frag) || !existsSync(js)) { warn(`figure ${arg} not built yet`); $p.remove(); return; }
      let inner = readFileSync(frag, "utf8");
      const wide = /^\s*<!--\s*wide\s*-->/.test(inner);
      inner = renderInlineDelims(inner.replace(/^\s*<!--\s*wide\s*-->\s*/, ""));
      figureCounter += 1;
      inner = inner.replace(/<figcaption>/, `<figcaption><span class="fig__num">Figure ${figureCounter}.</span>`);
      figures.push(arg);
      $p.replaceWith(`<figure class="fig${wide ? " fig--wide" : ""}" id="fig-${arg}">${inner}</figure>`);
    }
  });

  // \paragraph{X} arrives as h5; run it into the next paragraph like LaTeX does
  $("h5").each((_, h) => {
    const $h = $(h);
    const next = $h.next();
    const lead = `<strong class="para-head">${$h.html()}</strong> `;
    if (next.is("p")) next.prepend(lead); else $h.after(`<p>${lead}</p>`);
    $h.remove();
  });

  // amsthm environments: pandoc writes <div class="definition"> with a label span
  for (const kind of ["definition", "remark", "problem", "theorem", "lemma", "proof", "example"]) {
    $(`div.${kind}`).each((_, d) => {
      const $d = $(d);
      $d.attr("class", `thm thm--${kind}`);
      const first = $d.children("p").first();
      const label = first.children("strong").first();
      if (!label.length) return;
      label.addClass("thm__label");
      // \newtheorem* environments are unnumbered in the PDF; pandoc numbers them anyway
      if (opts.starred && opts.starred.has(kind)) label.text(label.text().replace(/\s+\d+$/, ""));
      // "(Name). " after the label joins it, as in LaTeX's bold-then-name heading
      const after = label[0].next;
      if (after && after.type === "text") {
        const m = after.data.match(/^\s*\(([^)]*)\)\.?\s*/);
        if (m) { after.data = after.data.slice(m[0].length); label.append(` <span class="thm__name">(${esc(m[1])})</span>`); }
        else after.data = after.data.replace(/^\.?\s*/, "");
      }
      label.append(". ");
      // the body arrives italic (amsthm's plain style); upright reads better on screen
      $d.find("> p > em").each((_, em) => {
        const $em = $(em);
        if ($em.parent().children().length <= 3) $em.replaceWith($em.html());
      });
    });
  }

  // tables scroll on their own instead of the page
  $("table").each((_, t) => { if (!$(t).parent().hasClass("table-wrap")) $(t).wrap('<div class="table-wrap"></div>'); });

  // pandoc's per-line anchors in code blocks
  $("div.sourceCode a[aria-hidden]").remove();
  $("pre a[href^='#cb']").remove();

  // empty paragraphs left by removed commands
  $("p").each((_, p) => { if (!$(p).text().trim() && !$(p).children().length) $(p).remove(); });

  // headings: number them and collect the contents list
  const toc = [];
  const counters = [0, 0, 0];
  $("h2, h3, h4").each((_, h) => {
    const $h = $(h);
    const level = Number(h.name[1]) - 2;
    if (!$h.attr("id")) $h.attr("id", slugify($h.text()));
    const unnumbered = $h.hasClass("unnumbered") || opts.unnumbered;
    let num = "";
    if (!unnumbered) {
      counters[level] += 1;
      for (let i = level + 1; i < counters.length; i++) counters[i] = 0;
      num = [opts.chapter, ...counters.slice(0, level + 1)].filter((x) => x !== undefined && x !== null).join(".");
      $h.prepend(`<span class="num">${num}</span>`);
    }
    if (level <= 1) toc.push({ level, id: $h.attr("id"), num, html: $h.clone().find(".num").remove().end().html() });
  });

  const words = $.root().text().split(/\s+/).filter(Boolean).length;
  return { html: $.html(), toc, figures, words };
}

function starredTheorems(preamble) {
  return new Set([...preamble.matchAll(/\\newtheorem\*\{(\w+)\}/g)].map((m) => m[1]));
}

function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// ---------------------------------------------------------------------------
// 4. templates

const THEME_INIT = `<script>(function(){try{var t=localStorage.getItem('theme');if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>`;
const ICONS = `<svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 1.6v2.4M12 20v2.4M4.2 4.2l1.7 1.7M18.1 18.1l1.7 1.7M1.6 12h2.4M20 12h2.4M4.2 19.8l1.7-1.7M18.1 5.9l1.7-1.7"/></svg>
      <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.3A8.6 8.6 0 0 1 9.7 3.5a8.6 8.6 0 1 0 10.8 10.8z"/></svg>`;

function page({ title, description, url, root, side = "", article, figures = [], ogType = "article", math = true }) {
  const scripts = [`<script src="${root}assets/js/main.js"></script>`, `<script src="${root}assets/js/writing.js"></script>`];
  if (figures.length) {
    scripts.push(`<script src="${root}assets/js/figlib.js"></script>`);
    for (const f of figures) scripts.push(`<script src="${root}assets/js/figures/${f}.js" defer></script>`);
  }
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(description)}">
<link rel="canonical" href="${url}">
<meta property="og:type" content="${ogType}">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:url" content="${url}">
<meta name="twitter:card" content="summary">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.92em' font-size='92' font-family='Georgia,serif' font-weight='600'>J</text></svg>">
<link rel="alternate" type="application/atom+xml" title="Jai Sharma: Writing" href="${root}writing/index.xml">
<link rel="stylesheet" href="${root}assets/css/style.css">
${math ? `<link rel="stylesheet" href="${root}assets/katex/katex.min.css">\n` : ""}<link rel="stylesheet" href="${root}assets/css/writing.css">
${THEME_INIT}
</head>
<body>

<div class="topbar">
  <div class="wrap topbar__inner topbar__inner--wide">
    <a class="home" href="${root}index.html">Jai Sharma</a>
    <nav>
      <a href="${root}index.html#research">Research</a>
      <a href="${root}writing/">Writing</a>
      <a href="${root}assets/pdf/Jai_Sharma_CV.pdf">CV</a>
    </nav>
    <button class="theme-toggle" type="button" aria-label="Switch theme">
      ${ICONS}
    </button>
  </div>
</div>

<div class="page">
${side}
<article class="article">
${article}
</article>
</div>

<footer class="wrap foot">
  <span>© ${new Date().getFullYear()} Jai Sharma</span>
  <span class="spacer"></span>
  <span>Math typeset with <a href="https://katex.org/">KaTeX</a>.</span>
</footer>

${scripts.join("\n")}
</body>
</html>
`;
}

function tocList(toc, { activeFirst = false } = {}) {
  let out = "<ol>";
  let open = false;
  toc.forEach((t, i) => {
    if (t.level === 0) {
      if (open) { out += "</ol></li>"; open = false; }
      if (i > 0) out += "</li>";
      out += `<li><a href="#${t.id}">${t.num ? `<span class="num">${t.num}</span>` : ""}<span>${t.html}</span></a>`;
      if (toc[i + 1] && toc[i + 1].level === 1) { out += "<ol>"; open = true; }
    } else {
      out += `<li><a href="#${t.id}">${t.num ? `<span class="num">${t.num}</span>` : ""}<span>${t.html}</span></a></li>`;
    }
  });
  if (open) out += "</ol>";
  if (toc.length) out += "</li>";
  return out + "</ol>";
}

function authorsText(a) {
  if (a.length <= 2) return a.join(" and ");
  return a.slice(0, -1).join(", ") + ", and " + a[a.length - 1];
}

function bibtex(doc, url) {
  const [y, m] = doc.date.split("-").map(Number);
  const fields = [
    ["title", doc.title],
    ["author", doc.authors.map((n) => { const p = n.split(" "); return `${p.slice(-1)[0]}, ${p.slice(0, -1).join(" ")}`; }).join(" and ")],
    doc.bibtype === "book" ? ["publisher", "Independently published"] : doc.bibtype === "techreport" ? ["institution", "UC Berkeley"] : ["journal", "jaisharmz.github.io"],
    doc.isbn ? ["isbn", doc.isbn] : null,
    ["year", String(y)],
    ["month", MONTHS[m - 1].slice(0, 3)],
    ["url", url]
  ].filter(Boolean);
  const w = Math.max(...fields.map(([k]) => k.length));
  return `@${doc.bibtype}{${doc.bibkey},\n${fields.map(([k, v]) => `  ${k.padEnd(w)} = "${v}"`).join(",\n")}\n}`;
}

function citeBlock(doc, url) {
  const who = doc.authors.length > 2 ? `${doc.authors[0].split(" ").slice(-1)[0]} et al.` : authorsText(doc.authors.map((n) => n.split(" ").slice(-1)[0]));
  return `<section class="cite">
  <h2>Cite this</h2>
  <p>${esc(who)} (${year(doc.date)}). ${esc(doc.title)}. <em>jaisharmz.github.io</em>.</p>
  <pre>${esc(bibtex(doc, url))}</pre>
</section>`;
}

const fmtDur = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const lectureList = (lectures, label = "Lectures for this chapter") => lectures && lectures.length ? `<div class="lectures">
  <p class="lectures__label">${label}</p>
  <ol>${lectures.map(([id, t, d]) => `<li><a href="https://www.youtube.com/watch?v=${id}&amp;list=PLVjRK697Kk5sFVpUdUD-veuNveDoi9Y1m">${esc(t)}<span class="dur">${fmtDur(d)}</span></a></li>`).join("")}</ol>
</div>` : "";

// ---------------------------------------------------------------------------
// documents

function buildPost(doc) {
  console.log(`post  ${doc.slug}`);
  figureCounter = 0;
  const tex = readFileSync(path.join(SRC, doc.src), "utf8");
  const { preamble, body } = splitDoc(tex);
  doc.starred = starredTheorems(preamble);
  const html = pandoc(latexPass(body, preamble, doc), preamble);
  const res = htmlPass(html, doc, {});
  const root = "../../";
  const url = `${config.site}/writing/${doc.slug}/`;
  const minutes = Math.max(1, Math.round(res.words / 230));
  const meta = [
    `<time datetime="${doc.date}">${longDate(doc.date)}</time>`,
    `<span>${minutes} min read</span>`,
    `<span>${esc(authorsText(doc.authors))}</span>`,
    doc.status ? `<span>${esc(doc.status)}</span>` : "",
    `<a href="${root}${doc.pdf}">PDF</a>`,
    ...(doc.extra || []).map((x) => `<a href="${root}${x.href}">${esc(x.label)}</a>`)
  ].filter(Boolean).join("\n    ");
  const side = res.toc.length ? `<aside class="side toc" aria-label="Contents">
  <p class="side__label">Contents</p>
  ${tocList(res.toc)}
</aside>` : "";
  const article = `<header class="article__head">
  <p class="article__kicker"><a href="${root}writing/">Writing</a>${doc.venue ? ` · ${esc(doc.venue)}` : ""}</p>
  <h1>${esc(doc.title)}</h1>
  <p class="article__dek">${esc(doc.dek)}</p>
  <p class="article__meta">
    ${meta}
  </p>
</header>
${res.toc.length ? `<details class="toc-inline"><summary>Contents</summary>${tocList(res.toc)}</details>` : ""}
<div class="prose-long">
${res.html.replace(/@ROOT@/g, root)}
</div>
${citeBlock(doc, url)}
<div class="web-note"><p>This page is rendered from the LaTeX source of the <a href="${root}${doc.pdf}">PDF</a>, which stays at its original address.${res.figures.length ? " The interactive figures are new to the web edition; each one runs the math in the text on a small example." : ""}</p></div>`;
  write(path.join(REPO, "writing", doc.slug, "index.html"), page({
    title: `${doc.title} | Jai Sharma`, description: doc.dek, url, root, side, article, figures: res.figures
  }));
  return { ...doc, minutes, url: `writing/${doc.slug}/`, figures: res.figures.length };
}

// Split a book's body into parts and sections, keeping each unit's own LaTeX.
function bookUnits(body) {
  const units = [];
  let part = null, cur = null;
  const re = /\\part\*?\{([^}]*)\}|\\section(\*?)\{([^}]*)\}/g;
  let last = 0, m;
  const flush = (end) => {
    const chunk = body.slice(last, end);
    if (cur) cur.tex += chunk;
    else if (part) part.intro += chunk;
  };
  while ((m = re.exec(body))) {
    flush(m.index);
    last = m.index + m[0].length;
    if (m[1] !== undefined && m[0].startsWith("\\part")) {
      part = { title: m[1], intro: "" };
      cur = null;
      units.push({ kind: "part", part });
    } else {
      cur = { kind: "section", part: part && part.title, title: m[3], star: m[2] === "*", tex: "" };
      units.push(cur);
    }
  }
  flush(body.length);
  return units;
}

function buildBook(doc) {
  console.log(`book  ${doc.slug}`);
  const tex = readFileSync(path.join(SRC, doc.src), "utf8");
  const { preamble, body } = splitDoc(tex);
  doc.starred = starredTheorems(preamble);
  const units = bookUnits(body);
  const parts = units.filter((u) => u.kind === "part").map((u) => u.part);
  const sections = units.filter((u) => u.kind === "section");
  const bookRoot = "../";
  const url = `${config.site}/writing/${doc.slug}/`;

  // assemble each chapter's LaTeX from its source sections
  const used = new Set();
  const chapters = doc.chapters.map((ch, i) => {
    const secs = ch.sections.map((title) => {
      const s = sections.find((x) => x.title === title && x.part === ch.part && !used.has(x));
      if (!s) throw new Error(`chapter ${ch.slug}: no section "${title}" in part "${ch.part}"`);
      used.add(s);
      return s;
    });
    const firstOfPart = doc.chapters.findIndex((c) => c.part === ch.part) === i;
    const intro = firstOfPart ? (parts.find((p) => p.title === ch.part) || { intro: "" }).intro : "";
    let chTex;
    if (secs.length === 1) {
      // one source section: its subsections become the chapter's sections
      chTex = intro + "\n\n" + secs[0].tex
        .replace(/\\subsubsection(\*?)\{/g, "\\XSUBSUB$1{")
        .replace(/\\subsection(\*?)\{/g, "\\section$1{")
        .replace(/\\XSUBSUB(\*?)\{/g, "\\subsection$1{");
    } else if (ch.leadIsTitle) {
      chTex = intro + "\n\n" + secs[0].tex + secs.slice(1).map((s) => `\n\n\\section{${s.title}}\n${s.tex}`).join("");
    } else {
      chTex = intro + secs.map((s) => `\n\n\\section{${s.title}}\n${s.tex}`).join("");
    }
    return { ...ch, title: ch.title || secs[0].title, tex: chTex };
  });
  const foreword = sections.find((s) => s.part === "Foreword");

  let n = 0;
  chapters.forEach((ch) => { if (!ch.unnumbered) ch.num = ++n; });

  // render chapters
  figureCounter = 0;
  const rendered = chapters.map((ch) => {
    const nb = ch.notebook ? loadNotebook(ch.notebook) : null;
    const html = pandoc(latexPass(ch.tex, preamble, doc), preamble);
    const res = htmlPass(html, doc, { chapter: ch.num, unnumbered: ch.unnumbered, notebook: nb });
    if (nb && nb.some((c) => !c.used)) {
      const left = nb.filter((c) => !c.used).length;
      if (left) console.log(`  ${ch.slug}: ${left} notebook outputs had no matching listing`);
    }
    return { ...ch, ...res };
  });

  const partsOrder = [...new Set(chapters.map((c) => c.part))];
  const sideFor = (current) => `<aside class="side toc book-toc" aria-label="Book contents">
  <p class="side__label"><a href="${current ? "../" : "./"}">${esc(doc.title)}</a></p>
  ${partsOrder.map((p) => `<p class="part">${esc(p)}</p>
  <ol>${rendered.filter((c) => c.part === p).map((c) => {
    const here = current && current.slug === c.slug;
    const href = current ? (here ? "./" : `../${c.slug}/`) : `${c.slug}/`;
    return `<li${here ? ' class="is-current"' : ""}><a href="${href}">${c.num ? `<span class="num">${c.num}</span>` : ""}<span>${esc(c.title)}</span></a>${here && c.toc.length ? tocList(c.toc.filter((t) => t.level === 0)) : ""}</li>`;
  }).join("")}</ol>`).join("\n  ")}
</aside>`;

  rendered.forEach((ch, i) => {
    const root = "../../../";
    const prev = rendered[i - 1], next = rendered[i + 1];
    const chUrl = `${url}${ch.slug}/`;
    const minutes = Math.max(1, Math.round(ch.words / 230));
    const nbLinks = ch.notebook ? ` · <a href="${doc.repo}${ch.notebook}">Notebook</a> · <a href="${doc.colab}${ch.notebook}">Open in Colab</a>` : "";
    const article = `<header class="article__head">
  <p class="article__kicker"><a href="../">${esc(doc.title)}</a> · ${esc(ch.part)}${ch.num ? ` · Chapter ${ch.num}` : ""}</p>
  <h1>${esc(ch.title)}</h1>
  <p class="article__meta">
    <span>${minutes} min read</span>
    <a href="${root}${doc.pdf}">PDF of the book</a>${nbLinks ? `\n    <span>${nbLinks.replace(/^ · /, "")}</span>` : ""}
  </p>
</header>
${lectureList(ch.lectures, ch.slug === "afterword" ? "The last lectures of the series" : "Lectures for this chapter")}
${ch.toc.length > 1 ? `<details class="toc-inline"><summary>In this chapter</summary>${tocList(ch.toc)}</details>` : ""}
<div class="prose-long">
${ch.html.replace(/@ROOT@/g, root)}
</div>
<nav class="endnav" aria-label="Chapters">
  ${prev ? `<a class="prev" href="../${prev.slug}/"><span class="dir">Previous</span><span class="t">${prev.num ? `${prev.num}. ` : ""}${esc(prev.title)}</span></a>` : `<a class="prev" href="../"><span class="dir">Contents</span><span class="t">${esc(doc.title)}</span></a>`}
  ${next ? `<a class="next" href="../${next.slug}/"><span class="dir">Next</span><span class="t">${next.num ? `${next.num}. ` : ""}${esc(next.title)}</span></a>` : `<a class="next" href="../"><span class="dir">Contents</span><span class="t">${esc(doc.title)}</span></a>`}
</nav>`;
    write(path.join(REPO, "writing", doc.slug, ch.slug, "index.html"), page({
      title: `${ch.title} | ${doc.title}`, description: ch.desc || doc.dek, url: chUrl, root,
      side: sideFor(ch), article, figures: ch.figures
    }));
  });

  // book home
  const root = "../../";
  const fw = foreword ? htmlPass(pandoc(latexPass(foreword.tex, preamble, doc), preamble), doc, { unnumbered: true }) : null;
  const totalMin = rendered.reduce((a, c) => a + Math.max(1, Math.round(c.words / 230)), 0);
  const lecCount = rendered.reduce((a, c) => a + (c.lectures || []).length, 0) + (doc.syllabus ? 1 : 0);
  const article = `<header class="article__head book-hero">
  <div>
    <p class="article__kicker"><a href="${root}writing/">Writing</a> · Book</p>
    <h1>${esc(doc.title)}</h1>
    <p class="article__dek">${esc(doc.dek)}</p>
    <p class="article__meta"><span>${esc(authorsText(doc.authors))}</span><span>${year(doc.date)}</span><span>ISBN ${doc.isbn}</span><span>${rendered.length - 1} chapters</span><span>${lecCount} lectures</span></p>
  </div>
  <a class="book-cover" href="${rendered[0].slug}/" aria-label="Start reading">
    <span class="book-cover__title">${esc(doc.title)}</span>
    ${doc.coverArt ? `<img src="${root}assets/img/writing/${doc.slug}/${doc.coverArt}" alt="">` : ""}
    <span class="book-cover__author">${esc(authorsText(doc.authors))}</span>
  </a>
</header>
<div class="btnrow">
  <a class="primary" href="${rendered[0].slug}/">Start reading</a>
  <a href="${root}${doc.pdf}">PDF</a>
  ${doc.links.map((l) => `<a href="${l.href}">${esc(l.label)}</a>`).join("\n  ")}
</div>
${doc.syllabus ? lectureList([[doc.syllabus.id, doc.syllabus.title, doc.syllabus.dur]], "Start with the syllabus") : ""}
${fw ? `<div class="prose-long"><h2 id="to-the-reader">To the Reader</h2>${fw.html.replace(/@ROOT@/g, root)}</div>` : ""}
<section class="contents" aria-label="Contents">
${partsOrder.map((p, k) => `<div class="contents__part">
  <h2>${p === "Afterword" ? "" : `Part ${["I", "II", "III", "IV", "V"][k]} · `}${esc(p)}</h2>
  <ol>${rendered.filter((c) => c.part === p).map((c) => `<li><a href="${c.slug}/"><span class="n">${c.num || ""}</span><span class="t">${esc(c.title)}</span><span class="l">${c.lectures ? `${c.lectures.length} lecture${c.lectures.length > 1 ? "s" : ""}` : ""}</span><span class="d">${esc(c.desc || "")}</span></a></li>`).join("")}</ol>
</div>`).join("\n")}
</section>
${citeBlock(doc, url)}
<div class="web-note"><p>This edition is rendered from the book's LaTeX source. The <a href="${root}${doc.pdf}">PDF</a> stays at its original address. Code outputs come from the companion notebooks, and the interactive figures are new to the web edition.</p></div>`;
  write(path.join(REPO, "writing", doc.slug, "index.html"), page({
    title: `${doc.title} | Jai Sharma`, description: doc.dek, url, root, side: sideFor(null), article, ogType: "book"
  }));
  return { ...doc, minutes: totalMin, url: `writing/${doc.slug}/`, chapters: rendered.length - 1, lectures: lecCount };
}

function buildIndex(entries) {
  const root = "../";
  const items = entries.slice().sort((a, b) => b.date.localeCompare(a.date)).map((e) => `<li>
    <span class="kind">${e.type === "book" ? `Book · ${e.chapters} chapters · ${e.lectures} lectures` : e.status ? `Post · ${esc(e.status)}` : e.venue ? "White paper" : "Post"}</span>
    <h2><a href="${e.slug}/">${esc(e.title)}</a></h2>
    <p>${esc(e.dek)}</p>
    <p class="meta"><time datetime="${e.date}">${longDate(e.date)}</time> · ${e.minutes} min read · ${esc(authorsText(e.authors))} · <a href="${root}${e.pdf}">PDF</a></p>
  </li>`).join("\n  ");
  const article = `<header class="article__head">
  <p class="article__kicker"><a href="${root}index.html">Jai Sharma</a></p>
  <h1>Writing</h1>
  <p class="article__dek">Long-form writing, rendered from the LaTeX sources. Every PDF stays at its original address. <a href="index.xml">RSS</a></p>
</header>
<ul class="wlist">
  ${items}
</ul>`;
  // an Atom feed, so readers can follow new posts
  const updated = entries.map((e) => e.date).sort().slice(-1)[0];
  write(path.join(REPO, "writing", "index.xml"), `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Jai Sharma: Writing</title>
  <link href="${config.site}/writing/"/>
  <link rel="self" href="${config.site}/writing/index.xml"/>
  <id>${config.site}/writing/</id>
  <updated>${updated}T00:00:00Z</updated>
  <author><name>${esc(config.author)}</name></author>
${entries.slice().sort((a, b) => b.date.localeCompare(a.date)).map((e) => `  <entry>
    <title>${esc(e.title)}</title>
    <link href="${config.site}/${e.url}"/>
    <id>${config.site}/${e.url}</id>
    <updated>${e.date}T00:00:00Z</updated>
    <summary>${esc(e.dek)}</summary>
  </entry>`).join("\n")}
</feed>
`);
  write(path.join(REPO, "writing", "index.html"), page({
    title: "Writing | Jai Sharma", description: "Long-form writing by Jai Sharma: diffusion from first principles, an introduction to deep learning, and a compute exchange white paper.",
    url: `${config.site}/writing/`, root, article, ogType: "website", math: false
  }));
}

// KaTeX's stylesheet and woff2 fonts, vendored so pages load nothing from elsewhere
function vendorKatex() {
  const dist = path.join(TOOLS, "node_modules", "katex", "dist");
  const out = path.join(REPO, "assets", "katex");
  mkdir(path.join(out, "fonts"));
  let css = readFileSync(path.join(dist, "katex.min.css"), "utf8");
  // only woff2 ships; every current browser takes it first
  css = css.replace(/,url\(fonts\/[^)]+\.woff\) format\("woff"\)|,url\(fonts\/[^)]+\.ttf\) format\("truetype"\)/g, "");
  writeFileSync(path.join(out, "katex.min.css"), css);
  for (const f of readdirSync(path.join(dist, "fonts"))) if (f.endsWith(".woff2")) copyFileSync(path.join(dist, "fonts", f), path.join(out, "fonts", f));
}

// ---------------------------------------------------------------------------

vendorKatex();
rmSync(path.join(REPO, "assets", "img", "writing"), { recursive: true, force: true });
const entries = config.docs.map((d) => (d.type === "book" ? buildBook(d) : buildPost(d)));
buildIndex(entries);
console.log(warnings.length ? `\n${warnings.length} warnings` : "\nno warnings");
