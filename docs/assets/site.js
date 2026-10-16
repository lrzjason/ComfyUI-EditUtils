/* ComfyUI-EditUtils blog behaviour.
   Two jobs only: reveal sections on scroll, and fill the measured class axis.

   The reveal is a decoration, never a gate: if the observer does not fire
   (reduced motion, no IntersectionObserver, a print or headless render) the
   content is shown outright, and a timer guarantees the same after 1.5s even
   if the observer was created but never produced an entry. */

(function(){
  var items = Array.prototype.slice.call(document.querySelectorAll('.rv'));
  var rows = Array.prototype.slice.call(document.querySelectorAll('#axis li'));

  function fillAxis(li){
    var area = parseFloat(li.getAttribute('data-area')) || 0;
    var fill = li.querySelector('.fill');
    if (!fill) return;
    var pct = Math.max(7, Math.min(100, (area / 16.5) * 100));
    fill.style.width = pct.toFixed(1) + '%';
  }

  function fillAll(){
    rows.forEach(function(li){
      window.requestAnimationFrame(function(){ fillAxis(li); });
    });
  }

  var shown = false;
  function showAll(){
    if (shown) return;
    shown = true;
    items.forEach(function(el){ el.classList.add('in'); });
    fillAll();
  }

  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reduce || !('IntersectionObserver' in window)) {
    showAll();
    return;
  }

  function observe(list, options, onHit){
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(entry){
        if (!entry.isIntersecting) return;
        onHit(entry.target);
        io.unobserve(entry.target);
      });
    }, options);
    list.forEach(function(el){ io.observe(el); });
  }

  var axis = document.getElementById('axis');
  if (axis) {
    observe([axis], { threshold: 0.2 }, fillAll);
  } else {
    fillAll();
  }

  observe(items, { rootMargin: '0px 0px -12% 0px', threshold: 0.15 }, function(el){
    el.classList.add('in');
    fillAll();
  });

  // Fail-safe: never leave content hidden behind a decoration.
  window.setTimeout(showAll, 1500);
})();
