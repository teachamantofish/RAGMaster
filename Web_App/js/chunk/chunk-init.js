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
const DEFAULT_CONFIG_URL = './config/defaults/chunkerconfig.py';
const LOG_POLL_INTERVAL_MS = 1200;
const LOG_TAIL_LINES = 120;

async function fetchLogTail(script, lines = LOG_TAIL_LINES) {
  const url = `/api/log-tail?script=${encodeURIComponent(script)}&lines=${encodeURIComponent(String(lines))}`;
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '');
    throw new Error(`Log tail failed: HTTP ${resp.status}${detail ? ` - ${detail}` : ''}`);
  }
  const result = await resp.json();
  if (!result || result.success !== true) {
    throw new Error(result?.error || 'Unknown log tail error.');
  }
  return result;
}

function renderRunningLog(statusEl, script, tailPayload) {
  if (!statusEl) return;
  const logBody = tailPayload && tailPayload.exists
    ? (tailPayload.output || '(log file exists but currently empty)')
    : '(waiting for log file to be created)';
  statusEl.value = `Running ${script}...\n\n${logBody}`;
}

function renderRunningLogToViewer(rawViewer, script, tailPayload) {
  if (!rawViewer) return;
  const logBody = tailPayload && tailPayload.exists
    ? (tailPayload.output || '(log file exists but currently empty)')
    : '(waiting for log file to be created)';
  rawViewer.value = `# Live log: ${script}\n\n${logBody}`;
}

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
  const rawViewer = container.querySelector('#chunk-chunkconfig');
  const defaultsBtn = ensureDefaultsButton(container);

  const isDirty = () => {
    if (!binder || !schema || !Array.isArray(schema.fields)) return false;
    try {
      const patch = binder.collectValues();
      const candidate = writeConfig(originalText, patch, schema.fields);
      return candidate !== originalText;
    } catch (_) {
      return false;
    }
  };

  if (reloadBtn) reloadBtn.title = 'Revert unsaved changes';

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
      if (isDirty() && !window.confirm('Discard unsaved changes and revert from disk?')) {
        return;
      }
      try {
        originalText = await loadConfigText();
        const parsed = parseConfig(originalText);
        model = parsed.values;

        // Re-bind by re-initializing
        binder = bindConfig(container, schema, model, showHelp);

        const rawViewer = container.querySelector('#chunk-chunkconfig');
        if (rawViewer) rawViewer.value = originalText;

        if (statusEl) statusEl.value = 'Changes reverted from disk.';
      } catch (err) {
        if (statusEl) statusEl.value = `Reload failed: ${err.message}`;
      }
    });
  }

  if (defaultsBtn) {
    defaultsBtn.addEventListener('click', async () => {
      if (!window.confirm('Reset this tab to saved defaults? This will overwrite current settings.')) {
        return;
      }
      try {
        const resp = await fetch(DEFAULT_CONFIG_URL, { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status} while loading defaults`);
        const defaultText = await resp.text();

        await saveConfigText(defaultText);
        originalText = defaultText;
        const parsed = parseConfig(defaultText);
        model = parsed.values;
        binder = bindConfig(container, schema, model, showHelp);

        const rawViewer = container.querySelector('#chunk-chunkconfig');
        if (rawViewer) rawViewer.value = defaultText;

        if (statusEl) statusEl.value = 'Configuration reset to defaults.';
      } catch (err) {
        if (statusEl) statusEl.value = `Reset to defaults failed: ${err.message}`;
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

      let pollTimer = null;
      let lastTail = null;
      let pollFailedAtLeastOnce = false;

      const pollOnce = async () => {
        try {
          const tail = await fetchLogTail(scriptPath, LOG_TAIL_LINES);
          lastTail = tail;
          renderRunningLog(statusEl, scriptPath, tail);
          renderRunningLogToViewer(rawViewer, scriptPath, tail);
        } catch (e) {
          pollFailedAtLeastOnce = true;
          if (statusEl) {
            statusEl.value = `Running ${scriptPath}...\n\nLive log unavailable right now (${e?.message || 'poll failed'}).`;
          }
        }
      };

      try {
        runBtn.disabled = true;
        runBtn.loading = true;
        if (statusEl) statusEl.value = `Running ${scriptPath}…`;

        await pollOnce();
        pollTimer = window.setInterval(pollOnce, LOG_POLL_INTERVAL_MS);

        const resp = await fetch('/api/run-script', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ script: scriptPath }),
        });
        const result = await resp.json();

        await pollOnce();

        const finalLogText = lastTail && lastTail.exists
          ? (lastTail.output || '(log file exists but currently empty)')
          : '';

        if (result.success) {
          if (statusEl) {
            statusEl.value = `✓ ${scriptPath} completed successfully.\n\n${finalLogText || result.output || ''}`;
          }
          if (rawViewer) {
            rawViewer.value = finalLogText
              ? `# Live log: ${scriptPath}\n\n${finalLogText}`
              : (result.output || rawViewer.value);
          }
        } else {
          if (statusEl) {
            const failureOutput = [finalLogText, result.output, result.error].filter(Boolean).join('\n\n');
            statusEl.value = `✗ ${scriptPath} failed (exit code ${result.exitCode}).\n\n${failureOutput}`;
          }
          if (rawViewer) {
            const failureOutput = [finalLogText, result.output, result.error].filter(Boolean).join('\n\n');
            rawViewer.value = `# Live log: ${scriptPath}\n\n${failureOutput}`;
          }
        }
      } catch (err) {
        if (statusEl) statusEl.value = `Run failed: ${err.message}`;
        if (rawViewer) {
          rawViewer.value = `# Live log: ${scriptPath}\n\nRun failed: ${err.message}`;
        }
      } finally {
        if (pollTimer !== null) {
          window.clearInterval(pollTimer);
        }
        if (pollFailedAtLeastOnce && !lastTail && rawViewer) {
          rawViewer.value = `${rawViewer.value}\n\n(Tip: restart Vite dev server so /api/log-tail is active.)`;
        }
        runBtn.disabled = false;
        runBtn.loading = false;
      }
    });
  }
}

function ensureDefaultsButton(container) {
  let btn = container.querySelector('#chunk-defaults-btn');
  if (btn) return btn;

  const actions = container.querySelector('.config-action-buttons');
  if (!actions) return null;

  btn = document.createElement('sl-button');
  btn.variant = 'primary';
  btn.id = 'chunk-defaults-btn';
  btn.title = 'Reset configuration to saved defaults';
  const icon = document.createElement('sl-icon');
  icon.name = 'arrow-counterclockwise';
  btn.appendChild(icon);
  actions.appendChild(btn);
  return btn;
}
