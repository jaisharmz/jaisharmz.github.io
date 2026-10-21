/* Writing pages: highlight the contents entry for the section being read. */
(function () {
  var links = document.querySelectorAll('.toc a[href^="#"]');
  if (!links.length || !window.IntersectionObserver) return;
  var byId = {};
  links.forEach(function (a) { byId[decodeURIComponent(a.getAttribute("href").slice(1))] = a; });
  var heads = Array.prototype.filter.call(
    document.querySelectorAll(".prose-long h2[id], .prose-long h3[id]"),
    function (h) { return byId[h.id]; }
  );
  if (!heads.length) return;

  var current = null;
  function set(id) {
    if (id === current) return;
    if (current && byId[current]) byId[current].classList.remove("is-active");
    current = id;
    var a = byId[id];
    if (!a) return;
    a.classList.add("is-active");
    // keep the active entry visible inside a scrolling sidebar
    var side = a.closest(".side");
    if (side && side.scrollHeight > side.clientHeight) {
      var r = a.getBoundingClientRect(), s = side.getBoundingClientRect();
      if (r.top < s.top + 40 || r.bottom > s.bottom - 40) side.scrollTop += r.top - s.top - s.height / 3;
    }
  }

  // the last heading above a line 30% down the viewport is the one being read
  function update() {
    var line = window.innerHeight * 0.3, id = heads[0].id;
    for (var i = 0; i < heads.length; i++) {
      if (heads[i].getBoundingClientRect().top <= line) id = heads[i].id; else break;
    }
    set(id);
  }
  var ticking = false;
  window.addEventListener("scroll", function () {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { ticking = false; update(); });
  }, { passive: true });
  window.addEventListener("resize", update);
  update();
})();
