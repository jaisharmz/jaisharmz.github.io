/* Theme toggle. The inline snippet in <head> sets the initial attribute so
   there is no flash; this only wires up the button and persists the choice. */
(function () {
  var root = document.documentElement;

  function current() {
    var set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function apply(mode) {
    root.setAttribute('data-theme', mode);
    try { localStorage.setItem('theme', mode); } catch (e) {}
    var btn = document.querySelector('.theme-toggle');
    if (btn) btn.setAttribute('aria-label', 'Switch to ' + (mode === 'dark' ? 'light' : 'dark') + ' theme');
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('.theme-toggle');
    if (!btn) return;
    apply(current() === 'dark' ? 'light' : 'dark');
  });

  apply(current());
})();
