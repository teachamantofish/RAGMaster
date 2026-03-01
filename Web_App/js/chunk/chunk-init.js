/**
 * chunk-init.js
 *
 * Chunk tab lifecycle controller.
 * Called after the chunk.html fragment is injected into the DOM.
 *
 * No pywebview dependency — all I/O uses standard HTTP fetch.
 * Load: GET  ./Get_and_Chunk/config/chunkerconfig.py
 * Save: PUT  ./Get_and_Chunk/config/chunkerconfig.py
 */

import { parseConfig } from '../config-parser.js';
import { writeConfig } from '../config-writer.js';
import { bindConfig } from '../config-binder.js';
import { showHelp } from '../help-modal.js';

const SCHEMA_URL = './config/chunk.schema.json';
const CONFIG_URL = './Get_and_Chunk/config/chunkerconfig.py';

/** @type {{ collectValues: Function, validate: Function }|null} */
let binder = null;
let originalText = '';
let schema = null;
let model = {};

/**
 * Main initializer — called once after chunk fragment is injected.
 * @param {HTMLElement} container - The #tab-content-chunk element.
 */
export async function initChunkTab(container) {
  try {
    // 1. Load schema
    schema = await fetchSchema();

    // 2. Load config file text
    originalText = await loadConfigText();

    // 3. Parse into typed model
    const parsed = parseConfig(originalText);
    model = parsed.values;

    // 4. Bind model to DOM controls
    binder = bindConfig(container, schema, model, showHelp);

    // 5. Also populate the raw file viewer textarea if present
    const rawViewer = container.querySelector('#chunk-chunkconfig');
    if (rawViewer) {
      rawViewer.value = originalText;
    }

    // 6. Wire action buttons
    wireButtons(container);

    console.log('[chunk-init] Chunk tab initialized successfully.');
  } catch (err) {
    console.error('[chunk-init] Failed to initialize chunk tab:', err);
    const status = container.querySelector('#chunk-status-display');
    if (status) {
      status.value = `Error initializing chunk tab: ${err.message}`;
    }
  }
}

/**
 * Fetch the chunk schema JSON.
 */
async function fetchSchema() {
  const resp = await fetch(SCHEMA_URL);
  if (!resp.ok) throw new Error(`Failed to load schema: HTTP ${resp.status}`);
  return resp.json();
}

/**
 * Load the config file text via HTTP GET.
 */
async function loadConfigText() {
  const resp = await fetch(CONFIG_URL);
  if (!resp.ok) throw new Error(`Failed to fetch config: HTTP ${resp.status}`);
  return resp.text();
}

/**
 * Save the config file text via HTTP PUT.
 */
async function saveConfigText(text) {
  const resp = await fetch(CONFIG_URL, {
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

/**
 * Wire save, reload, and run buttons.
 */
function wireButtons(container) {
  const saveBtn = container.querySelector('#chunk-save-btn');
  const reloadBtn = container.querySelector('#chunk-reload-btn');
  const runBtn = container.querySelector('#chunk-run-btn');
  const statusEl = container.querySelector('#chunk-status-display');

  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      if (!binder) return;
      const { valid, errors } = binder.validate();
      if (!valid) {
        const msg = errors.map(e => `• ${e.message}`).join('\n');
        if (statusEl) statusEl.value = `Validation errors:\n${msg}`;
        return;
      }

      try {
        const patch = binder.collectValues();
        const newText = writeConfig(originalText, patch, schema.fields);
        await saveConfigText(newText);
        originalText = newText;

        // Update raw viewer
        const rawViewer = container.querySelector('#chunk-chunkconfig');
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
        originalText = await loadConfigText();
        const parsed = parseConfig(originalText);
        model = parsed.values;

        // Re-bind by re-initializing
        binder = bindConfig(container, schema, model, showHelp);

        const rawViewer = container.querySelector('#chunk-chunkconfig');
        if (rawViewer) rawViewer.value = originalText;

        if (statusEl) statusEl.value = 'Configuration reloaded.';
      } catch (err) {
        if (statusEl) statusEl.value = `Reload failed: ${err.message}`;
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      const scriptPath = schema && schema.runScript;
      if (!scriptPath) {
        if (statusEl) statusEl.value = 'Run: no pipeline script configured for this tab.';
        return;
      }
      try {
        runBtn.disabled = true;
        runBtn.loading = true;
        if (statusEl) statusEl.value = `Running ${scriptPath}…`;

        const resp = await fetch('/api/run-script', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ script: scriptPath }),
        });
        const result = await resp.json();

        if (result.success) {
          if (statusEl) statusEl.value = `✓ ${scriptPath} completed successfully.\n\n${result.output || ''}`;
        } else {
          if (statusEl) statusEl.value = `✗ ${scriptPath} failed (exit code ${result.exitCode}).\n\n${result.output || result.error || ''}`;
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
