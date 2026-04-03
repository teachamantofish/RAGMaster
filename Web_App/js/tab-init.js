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
    const resolvedConfigUrl = toRootRelativeUrl(configUrl);

    try {
      // 1. Load schema
      const schemaResp = await fetch(schemaUrl);
      if (!schemaResp.ok) throw new Error(`Failed to load schema: HTTP ${schemaResp.status}`);
      schema = await schemaResp.json();
      runScript = schema.runScript || null;

      // 2. Load config file text
      const configResp = await fetch(resolvedConfigUrl);
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
        schema, resolvedConfigUrl, (b) => { binder = b; }, (m) => { model = m; }, runScript);

      console.log(`[${phase}-init] Tab initialized successfully.`);
    } catch (err) {
      console.error(`[${phase}-init] Failed to initialize tab:`, err);
      const status = container.querySelector(`#${prefix}-status-display`);
      if (status) status.value = `Error initializing ${phase} tab: ${err.message}`;
    }
  };
}

function toRootRelativeUrl(url) {
  const normalized = String(url || '').replace(/\\/g, '/').trim();
  if (!normalized) return normalized;
  if (/^(https?:)?\/\//i.test(normalized)) return normalized;
  if (normalized.startsWith('/')) return normalized;
  if (normalized.startsWith('./')) return `/${normalized.slice(2)}`;
  return `/${normalized}`;
}

/**
 * Wire save, reload, and run buttons.
 */
function wireButtons(container, prefix, getBinder, getOriginal, setOriginal, schema, configUrl, setBinder, setModel, runScript) {
  const saveBtn = container.querySelector(`#${prefix}-save-btn`);
  const reloadBtn = container.querySelector(`#${prefix}-reload-btn`);
  const runBtn = container.querySelector(`#${prefix}-run-btn`);
  const stopBtn = container.querySelector(`#${prefix}-stop-btn`);
  const statusEl = container.querySelector(`#${prefix}-status-display`);
  const rawViewer = container.querySelector(`#${prefix}-config-viewer`);
  const defaultsBtn = ensureDefaultsButton(container, prefix);
  let currentRunController = null;
  let currentRunScript = null;

  const isDirty = () => {
    const binder = getBinder();
    if (!binder || !schema || !Array.isArray(schema.fields)) return false;
    try {
      const patch = binder.collectValues();
      const candidate = writeConfig(getOriginal(), patch, schema.fields);
      return candidate !== getOriginal();
    } catch (_) {
      return false;
    }
  };

  if (reloadBtn) reloadBtn.title = 'Revert unsaved changes';

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
      if (isDirty() && !window.confirm('Discard unsaved changes and revert from disk?')) {
        return;
      }
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
        const defaultConfigUrl = buildDefaultConfigUrl(configUrl);
        const defaultResp = await fetch(defaultConfigUrl, { cache: 'no-store' });
        if (!defaultResp.ok) throw new Error(`HTTP ${defaultResp.status} while loading defaults`);
        const defaultText = await defaultResp.text();

        await saveConfigText(configUrl, defaultText);
        setOriginal(defaultText);

        const parsed = parseConfig(defaultText);
        setModel(parsed.values);
        setBinder(bindConfig(container, schema, parsed.values, showHelp));

        const rawViewer = container.querySelector(`#${prefix}-config-viewer`);
        if (rawViewer) rawViewer.value = defaultText;

        if (statusEl) statusEl.value = 'Configuration reset to defaults.';
      } catch (err) {
        if (statusEl) statusEl.value = `Reset to defaults failed: ${err.message}`;
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (!runScript) {
        if (statusEl) statusEl.value = 'Run: no pipeline script configured for this tab.';
        return;
      }

      let pollTimer = null;
      let lastTail = null;
      let pollFailedAtLeastOnce = false;
      const runController = new AbortController();
      currentRunController = runController;
      currentRunScript = runScript;

      const pollOnce = async () => {
        try {
          const tail = await fetchLogTail(runScript, LOG_TAIL_LINES);
          lastTail = tail;
          renderRunningLog(statusEl, runScript, tail);
          renderRunningLogToViewer(rawViewer, runScript, tail);
        } catch (e) {
          pollFailedAtLeastOnce = true;
          if (statusEl) {
            statusEl.value = `Running ${runScript}...\n\nLive log unavailable right now (${e?.message || 'poll failed'}).`;
          }
        }
      };

      try {
        runBtn.disabled = true;
        runBtn.loading = true;
        if (stopBtn) stopBtn.disabled = false;
        if (statusEl) statusEl.value = `Running ${runScript}…`;

        await pollOnce();
        pollTimer = window.setInterval(pollOnce, LOG_POLL_INTERVAL_MS);

        const resp = await fetch('/api/run-script', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: runController.signal,
          body: JSON.stringify({ script: runScript }),
        });
        const result = await resp.json();

        await pollOnce();

        const finalLogText = lastTail && lastTail.exists
          ? (lastTail.output || '(log file exists but currently empty)')
          : '';

        if (result.success) {
          if (statusEl) {
            statusEl.value = `✓ ${runScript} completed successfully.\n\n${finalLogText || result.output || ''}`;
          }
          if (rawViewer) {
            rawViewer.value = finalLogText
              ? `# Live log: ${runScript}\n\n${finalLogText}`
              : (result.output || rawViewer.value);
          }
        } else {
          if (statusEl) {
            const failureOutput = [finalLogText, result.output, result.error].filter(Boolean).join('\n\n');
            statusEl.value = `✗ ${runScript} failed (exit code ${result.exitCode}).\n\n${failureOutput}`;
          }
          if (rawViewer) {
            const failureOutput = [finalLogText, result.output, result.error].filter(Boolean).join('\n\n');
            rawViewer.value = `# Live log: ${runScript}\n\n${failureOutput}`;
          }
        }
      } catch (err) {
        const wasStopped = err?.name === 'AbortError';
        if (wasStopped) {
          if (statusEl) statusEl.value = `Stopped ${runScript}.`;
          if (rawViewer) {
            rawViewer.value = `# Live log: ${runScript}\n\nRun stopped by user.`;
          }
        } else {
          if (statusEl) statusEl.value = `Run failed: ${err.message}`;
          if (rawViewer) {
            rawViewer.value = `# Live log: ${runScript}\n\nRun failed: ${err.message}`;
          }
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
        if (stopBtn) stopBtn.disabled = true;
        if (currentRunController === runController) {
          currentRunController = null;
          currentRunScript = null;
        }
      }
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', async () => {
      if (!currentRunScript) return;
      stopBtn.disabled = true;
      if (statusEl) statusEl.value = `Stopping ${currentRunScript}...`;

      try {
        await fetch('/api/stop-script', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ script: currentRunScript }),
        });
      } catch (_) {
        // Keep stop flow best-effort: abort client-side request even if stop API fails.
      }

      if (currentRunController) {
        currentRunController.abort();
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

function buildDefaultConfigUrl(configUrl) {
  const normalized = String(configUrl || '').replace(/\\/g, '/');
  const fileName = normalized.split('/').filter(Boolean).pop() || '';
  if (!fileName) {
    throw new Error(`Cannot resolve defaults path for ${configUrl}`);
  }
  return `./config/defaults/${fileName}`;
}

function ensureDefaultsButton(container, prefix) {
  let btn = container.querySelector(`#${prefix}-defaults-btn`);
  if (btn) return btn;

  const actions = container.querySelector('.config-action-buttons');
  if (!actions) return null;

  btn = document.createElement('sl-button');
  btn.variant = 'primary';
  btn.id = `${prefix}-defaults-btn`;
  btn.title = 'Reset configuration to saved defaults';
  const icon = document.createElement('sl-icon');
  icon.name = 'arrow-counterclockwise';
  btn.appendChild(icon);
  actions.appendChild(btn);
  return btn;
}
