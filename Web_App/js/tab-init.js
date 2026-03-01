/**
 * tab-init.js
 *
 * Generic config-tab lifecycle factory.
 * Creates an initializer for any tab that follows the schema-driven
 * load → bind → save/reload pattern established by chunk-init.js.
 *
 * No pywebview dependency — all I/O uses standard HTTP fetch.
 */

import { parseConfig } from './config-parser.js';
import { writeConfig } from './config-writer.js';
import { bindConfig } from './config-binder.js';
import { showHelp } from './help-modal.js';

/**
 * Create a tab initializer.
 *
 * @param {object} opts
 * @param {string} opts.phase      - Tab name / phase id (e.g. 'crawl', 'clean').
 * @param {string} opts.schemaUrl  - URL to the schema JSON (e.g. './config/crawl.schema.json').
 * @param {string} opts.configUrl  - URL to the Python config file (e.g. './Get_and_Chunk/config/crawlconfig.py').
 * @param {string} opts.prefix     - HTML id prefix for controls (e.g. 'crawl').
 * @returns {(container: HTMLElement) => Promise<void>}
 */
export function createTabInit({ phase, schemaUrl, configUrl, prefix }) {
  return async function initTab(container) {
    let binder = null;
    let originalText = '';
    let schema = null;
    let model = {};
    let runScript = null;

    try {
      // 1. Load schema
      const schemaResp = await fetch(schemaUrl);
      if (!schemaResp.ok) throw new Error(`Failed to load schema: HTTP ${schemaResp.status}`);
      schema = await schemaResp.json();
      runScript = schema.runScript || null;

      // 2. Load config file text
      const configResp = await fetch(configUrl);
      if (!configResp.ok) throw new Error(`Failed to fetch config: HTTP ${configResp.status}`);
      originalText = await configResp.text();

      // 3. Parse into typed model
      const parsed = parseConfig(originalText);
      model = parsed.values;

      // 4. Bind model to DOM controls
      binder = bindConfig(container, schema, model, showHelp);

      // 5. Populate raw file viewer
      const rawViewer = container.querySelector(`#${prefix}-config-viewer`);
      if (rawViewer) rawViewer.value = originalText;

      // 6. Wire action buttons
      wireButtons(container, prefix, () => binder, () => originalText, (t) => { originalText = t; },
        schema, configUrl, (b) => { binder = b; }, (m) => { model = m; }, runScript);

      console.log(`[${phase}-init] Tab initialized successfully.`);
    } catch (err) {
      console.error(`[${phase}-init] Failed to initialize tab:`, err);
      const status = container.querySelector(`#${prefix}-status-display`);
      if (status) status.value = `Error initializing ${phase} tab: ${err.message}`;
    }
  };
}

/**
 * Wire save, reload, and run buttons.
 */
function wireButtons(container, prefix, getBinder, getOriginal, setOriginal, schema, configUrl, setBinder, setModel, runScript) {
  const saveBtn = container.querySelector(`#${prefix}-save-btn`);
  const reloadBtn = container.querySelector(`#${prefix}-reload-btn`);
  const runBtn = container.querySelector(`#${prefix}-run-btn`);
  const statusEl = container.querySelector(`#${prefix}-status-display`);

  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      const binder = getBinder();
      if (!binder) return;
      const { valid, errors } = binder.validate();
      if (!valid) {
        const msg = errors.map(e => `• ${e.message}`).join('\n');
        if (statusEl) statusEl.value = `Validation errors:\n${msg}`;
        return;
      }

      try {
        const patch = binder.collectValues();
        const newText = writeConfig(getOriginal(), patch, schema.fields);
        await saveConfigText(configUrl, newText);
        setOriginal(newText);

        const rawViewer = container.querySelector(`#${prefix}-config-viewer`);
        if (rawViewer) rawViewer.value = newText;

        if (statusEl) statusEl.value = 'Configuration saved successfully.';
      } catch (err) {
        if (statusEl) statusEl.value = `Save failed: ${err.message}`;
      }
    });
  }

  if (reloadBtn) {
    reloadBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch(configUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const text = await resp.text();
        setOriginal(text);

        const parsed = parseConfig(text);
        setModel(parsed.values);

        setBinder(bindConfig(container, schema, parsed.values, showHelp));

        const rawViewer = container.querySelector(`#${prefix}-config-viewer`);
        if (rawViewer) rawViewer.value = text;

        if (statusEl) statusEl.value = 'Configuration reloaded.';
      } catch (err) {
        if (statusEl) statusEl.value = `Reload failed: ${err.message}`;
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (!runScript) {
        if (statusEl) statusEl.value = 'Run: no pipeline script configured for this tab.';
        return;
      }
      try {
        runBtn.disabled = true;
        runBtn.loading = true;
        if (statusEl) statusEl.value = `Running ${runScript}…`;

        const resp = await fetch('/api/run-script', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ script: runScript }),
        });
        const result = await resp.json();

        if (result.success) {
          if (statusEl) statusEl.value = `✓ ${runScript} completed successfully.\n\n${result.output || ''}`;
        } else {
          if (statusEl) statusEl.value = `✗ ${runScript} failed (exit code ${result.exitCode}).\n\n${result.output || result.error || ''}`;
        }
      } catch (err) {
        if (statusEl) statusEl.value = `Run failed: ${err.message}`;
      } finally {
        runBtn.disabled = false;
        runBtn.loading = false;
      }
    });
  }
}

/**
 * Save config text via HTTP PUT.
 */
async function saveConfigText(url, text) {
  const resp = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    body: text,
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '');
    throw new Error(`Save failed: HTTP ${resp.status}${detail ? ' — ' + detail : ''}`);
  }
  const result = await resp.json().catch(() => null);
  if (result && !result.success) {
    throw new Error(result.error || 'Save failed');
  }
}
