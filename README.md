# jaisharmz.github.io

Personal academic site for Jai Sharma. Live at https://jaisharmz.github.io

Static HTML/CSS, no build step, no external assets. Layout follows the
Jon Barron academic template convention.

```
index.html            single page, everything lives here
assets/css/style.css  design system (light + dark, follows system or toggle)
assets/js/main.js     theme toggle only
assets/pdf/           hosted papers, books, white papers, CV
assets/img/thumbs/    publication teasers (rendered from PDF page 1)
```

## Editing

**Add a photo.** Drop a square image at `assets/img/profile.jpg`, then in
`index.html` replace `<span class="monogram">JS</span>` with
`<img src="assets/img/profile.jpg" alt="Jai Sharma">`.

**Add a publication.** Copy an `<li class="pub">` block in `#research`. Generate a
teaser with:

```sh
pdftoppm -jpeg -r 72 -f 1 -l 1 -scale-to-x 340 -scale-to-y -1 \
  assets/pdf/NAME.pdf assets/img/thumbs/NAME
```

**Add a news item.** Prepend an `<li>` to `.news` in `#news`.

Deployed from `main` via GitHub Pages.
