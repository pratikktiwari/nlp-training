(() => {
  const key = 'w2vtheme';

  function readTheme() {
    try {
      return localStorage.getItem(key) === 'dark' ? 'dark' : 'light';
    } catch (error) {
      console.warn('Theme storage is unavailable; keeping the current theme.', error);
      return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
    }
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const button = document.getElementById('bTheme');
    if (button) {
      const next = theme === 'dark' ? 'light' : 'dark';
      button.textContent = `${next === 'dark' ? 'Dark' : 'Light'} mode`;
      button.setAttribute('aria-label', `Switch to ${next} theme`);
      button.setAttribute('aria-pressed', String(theme === 'dark'));
      button.title = `Current theme: ${theme}`;
    }
  }

  applyTheme(readTheme());

  document.addEventListener('DOMContentLoaded', () => {
    applyTheme(document.documentElement.dataset.theme);
    document.getElementById('bTheme').addEventListener('click', () => {
      const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(theme);
      try {
        localStorage.setItem(key, theme);
      } catch (error) {
        console.warn('Theme changed, but the preference could not be saved.', error);
      }
    });
  });

  addEventListener('pageshow', () => applyTheme(readTheme()));
  addEventListener('storage', event => {
    if (event.key === key || event.key === null) {
      applyTheme(event.newValue === 'dark' ? 'dark' : 'light');
    }
  });
})();
