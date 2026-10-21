const PythonSyntax = (() => {
  const keywords = new Set([
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'case', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
    'match', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield',
  ]);
  const builtins = new Set([
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int',
    'isinstance', 'len', 'list', 'map', 'max', 'min', 'object', 'open',
    'print', 'range', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
    'sum', 'super', 'tuple', 'type', 'zip', 'Exception', 'RuntimeError',
    'TypeError', 'ValueError',
  ]);
  const escape = value => value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const tokens = /(?:[rubf]{1,2})?(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\[\s\S]|[^"\\\r\n])*"|'(?:\\[\s\S]|[^'\\\r\n])*')|#[^\r\n]*|(?:0x[\da-f_]+|0b[01_]+|0o[0-7_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:e[+-]?[\d_]+)?j?)|\b[a-z_]\w*\b|(?:\*\*|\/\/|:=|==|!=|<=|>=|[-+*/%@&|^~<>=!:]+)|[()[\]{},.;]/gi;

  function highlight(source) {
    let result = '', end = 0;
    for (const match of source.matchAll(tokens)) {
      const value = match[0], start = match.index;
      result += escape(source.slice(end, start));
      let kind;
      if (value.startsWith('#')) kind = 'comment';
      else if (/^(?:[rubf]{1,2})?['"]/i.test(value)) kind = 'string';
      else if (/^(?:\d|\.\d)/.test(value)) kind = 'number';
      else if (keywords.has(value)) kind = 'keyword';
      else if (builtins.has(value)) kind = 'builtin';
      else if (/^[a-z_]\w*$/i.test(value)) {
        if (/^\s*\(/.test(source.slice(start + value.length))) kind = 'function';
        else if (/^[A-Z][A-Z_\d]*$/.test(value)) kind = 'constant';
      } else kind = /^[()[\]{},.;]$/.test(value) ? 'punctuation' : 'operator';
      // Close token spans on each line so listings can keep their own row markup.
      result += kind ? value.split('\n').map(line =>
        line ? `<span class="py-token py-${kind}">${escape(line)}</span>` : '',
      ).join('\n') : escape(value);
      end = start + value.length;
    }
    return result + escape(source.slice(end));
  }

  function highlightLines(source) {
    return highlight(source).split('\n');
  }

  function highlightMarkup(markup) {
    const template = document.createElement('template');
    template.innerHTML = markup;
    const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const replacement = document.createElement('template');
      replacement.innerHTML = highlight(node.textContent);
      node.replaceWith(replacement.content);
    }
    return template.innerHTML;
  }

  const controlled = new WeakSet();
  const icons = {
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h4"/>',
    expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 3l6 6m12-6-6 6M3 21l6-6m12 6-6-6"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
  };

  function iconButton(label, icon) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'code-icon-button';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[icon]}</svg>`;
    return button;
  }

  function plainCode(source) {
    return Array.from(source.children, line => {
      const text = line.matches('.ln') ? line.querySelector('.t')
        : line.matches('.source-line') ? line.lastElementChild : null;
      return text ? text.textContent : '';
    }).join('\n');
  }

  function copyControls(source) {
    const tools = document.createElement('div');
    tools.className = 'code-tools';
    const feedback = document.createElement('span');
    feedback.className = 'code-feedback';
    feedback.setAttribute('role', 'status');
    const copy = iconButton('Copy code', 'copy');
    let timeout;
    copy.addEventListener('click', async () => {
      clearTimeout(timeout);
      try {
        if (!navigator.clipboard?.writeText) {
          throw new Error('Clipboard access requires HTTPS or localhost.');
        }
        await navigator.clipboard.writeText(plainCode(source));
        feedback.textContent = 'Copied';
        timeout = setTimeout(() => { feedback.textContent = ''; }, 2000);
      } catch (error) {
        console.error('Could not copy the Python code.', error);
        feedback.textContent = 'Could not copy. Select the code and copy manually.';
      }
    });
    tools.append(feedback, copy);
    return tools;
  }

  function attachControls(source) {
    if (controlled.has(source)) return;
    controlled.add(source);
    const panel = source.closest('.panel');
    const heading = panel.querySelector('h3');
    let header = panel.querySelector('.code-head');
    if (!header) {
      header = document.createElement('div');
      header.className = 'code-block-heading';
      heading.before(header);
      header.append(heading);
    }
    const tools = copyControls(source);
    const expand = iconButton('Expand code', 'expand');
    expand.setAttribute('aria-haspopup', 'dialog');
    tools.append(expand);
    header.append(tools);
    expand.addEventListener('click', () => {
      const dialog = document.createElement('dialog');
      dialog.className = 'code-dialog';
      dialog.setAttribute('aria-label', `${heading.textContent} — expanded code`);
      const bar = document.createElement('div');
      bar.className = 'code-dialog-heading';
      const title = document.createElement('h3');
      title.textContent = heading.textContent;
      const overlayTools = copyControls(source);
      const close = iconButton('Close expanded code', 'close');
      close.autofocus = true;
      close.addEventListener('click', dismiss);
      overlayTools.append(close);
      bar.append(title, overlayTools);
      const placeholder = document.createComment('Expanded code returns here.');
      const tabindex = source.getAttribute('tabindex');
      source.replaceWith(placeholder);
      source.tabIndex = 0;
      source.classList.add('code-expanded-source');
      dialog.append(bar, source);
      document.body.append(dialog);
      document.documentElement.classList.add('code-overlay-open');
      function restore() {
        if (!placeholder.parentNode) return;
        placeholder.replaceWith(source);
        if (tabindex === null) source.removeAttribute('tabindex');
        else source.setAttribute('tabindex', tabindex);
        source.classList.remove('code-expanded-source');
        document.documentElement.classList.remove('code-overlay-open');
        dialog.remove();
        expand.focus({ preventScroll: true });
      }
      function dismiss() {
        dialog.close();
        restore();
      }
      dialog.addEventListener('close', restore, { once: true });
      dialog.addEventListener('cancel', event => {
        event.preventDefault();
        dismiss();
      });
      dialog.addEventListener('keydown', event => {
        event.stopPropagation();
        if (event.key === 'Escape') {
          event.preventDefault();
          dismiss();
        }
      });
      dialog.showModal();
    });
  }

  return { highlight, highlightLines, highlightMarkup, attachControls };
})();
