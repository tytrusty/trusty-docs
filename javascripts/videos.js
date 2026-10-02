// Docs videos are muted. With motion allowed they play while on screen,
// without controls, like an animated image: a looping video pauses when
// scrolled away and resumes when back; a video without `loop` (the home-page
// teaser) plays once, the first time it is seen, and then stays on its last
// frame. With "reduce motion" set they stay paused on their poster and keep
// their controls. Without JS, the poster and controls from the HTML are all
// there is.
(function () {
  function setup() {
    var videos = document.querySelectorAll("video.shot");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    videos.forEach(function (v) { v.controls = false; });
    function play(v) {
      if (!v.loop && v.dataset.played) return;
      v.dataset.played = "1";
      v.play().catch(function () {});
    }
    if (!("IntersectionObserver" in window)) {
      videos.forEach(play);
      return;
    }
    var seen = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) play(e.target);
        else if (e.target.loop) e.target.pause();
      });
    }, { threshold: 0.25 });
    videos.forEach(function (v) { seen.observe(v); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setup);
  else setup();
})();
