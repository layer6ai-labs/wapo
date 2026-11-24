'use strict';

const progress = document.getElementById('progress');
function updateProgress() {
  const extent = document.documentElement.scrollHeight - innerHeight;
  progress.style.width = `${extent > 0 ? Math.min(100, 100 * scrollY / extent) : 0}%`;
}
addEventListener('scroll', updateProgress, { passive: true });
addEventListener('resize', updateProgress);
addEventListener('load', updateProgress);
document.querySelectorAll('details').forEach(panel => panel.addEventListener('toggle', updateProgress));

const links = [...document.querySelectorAll('.story-nav nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const first = entries.filter(entry => entry.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (first) links.forEach(link => {
      if (link.hash === `#${first.target.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-5% 0px -60% 0px' });
  document.querySelectorAll('article > section').forEach(section => observer.observe(section));
}
