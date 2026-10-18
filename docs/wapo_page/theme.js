'use strict';

(() => {
  const storageKey = 'wapo-theme';
  const root = document.documentElement;
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  let preference = null;

  try {
    const saved = window.localStorage.getItem(storageKey);
    if (saved === 'light' || saved === 'dark') preference = saved;
  } catch {
    // Private browsing and local-file previews may block preference storage.
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').setAttribute('content', theme === 'dark' ? '#101d17' : '#f5faf7');
    const toggle = document.getElementById('theme-toggle');
    if (toggle) toggle.setAttribute('aria-checked', String(theme === 'dark'));
  }

  // Run before styles load so a saved dark preference is applied on first paint.
  applyTheme(preference || (systemTheme.matches ? 'dark' : 'light'));

  systemTheme.addEventListener('change', event => {
    if (preference === null) applyTheme(event.matches ? 'dark' : 'light');
  });

  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('theme-toggle');
    toggle.setAttribute('aria-checked', String(root.dataset.theme === 'dark'));
    toggle.hidden = false;
    toggle.addEventListener('click', () => {
      preference = root.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(preference);
      try {
        window.localStorage.setItem(storageKey, preference);
      } catch {
        // The switch remains usable for this visit when storage is unavailable.
      }
    });
  }, { once: true });
})();
