/**
 * schema-coverage.test.js
 *
 * Ensures every config key in chunkerconfig.py is covered by the schema,
 * every schema field maps to a DOM control, and round-trip write preserves
 * comments and unmodified values.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { parseConfig, serializeValue } from '../../../js/config-parser.js';
import { writeConfig } from '../../../js/config-writer.js';

// ---------- Paths ----------
// Vitest may run from Web_App or Web_App/vite depending on invocation.
// Try both to find the schema file.
function findWebAppRoot() {
  // If cwd is Web_App, config/ is right here
  let candidate = process.cwd();
  if (existsSync(resolve(candidate, 'config/chunk.schema.json'))) return candidate;
  // If cwd is Web_App/vite, go up one level
  candidate = resolve(process.cwd(), '..');
  if (existsSync(resolve(candidate, 'config/chunk.schema.json'))) return candidate;
  return process.cwd(); // fallback
}

const webAppRoot = findWebAppRoot();
const schemaPath = resolve(webAppRoot, 'config/chunk.schema.json');
const configPath = resolve(webAppRoot, '..', 'Get_and_Chunk/config/chunkerconfig.py');

// ---------- Load fixtures ----------
let schema, configText, parsed;

try {
  schema = JSON.parse(readFileSync(schemaPath, 'utf8'));
} catch (e) {
  console.warn(`[schema-coverage] Could not load schema: ${schemaPath}`, e.message);
}

try {
  configText = readFileSync(configPath, 'utf8');
  parsed = parseConfig(configText);
} catch (e) {
  console.warn(`[schema-coverage] Could not load config: ${configPath}`, e.message);
}

// ---------- Fragment HTML (mirrors chunk.html controls) ----------
const CHUNK_HTML = `
  <div class="config-field"><sl-input data-config-key="TOKENIZER"></sl-input></div>
  <div class="config-field"><sl-input data-config-key="MAX_TOKENS_FOR_NODE"></sl-input></div>
  <div class="config-field"><sl-input data-config-key="CHUNK_SIZE_RANGE"></sl-input></div>
  <div class="config-field"><sl-select data-config-key="CHUNK_MODEL"></sl-select></div>
  <div class="config-field"><sl-input data-config-key="CODE_LENGTH"></sl-input></div>
  <div class="config-field"><sl-input data-config-key="KEYWORD_DENSITY"></sl-input></div>
  <div class="config-field"><sl-radio-group data-config-key="ENABLE_CODE_EXTRACTION"></sl-radio-group></div>
  <div class="config-field"><sl-input data-config-key="OUTPUT_NAME"></sl-input></div>
`;

describe('Schema coverage', () => {
  it('schema file loads correctly', () => {
    expect(schema).toBeDefined();
    expect(schema.fields).toBeDefined();
    expect(schema.fields.length).toBeGreaterThan(0);
  });

  it('every config key is represented in the schema', () => {
    if (!parsed || !schema) return;
    const schemaKeys = new Set(schema.fields.map(f => f.key));
    for (const key of Object.keys(parsed.values)) {
      expect(schemaKeys.has(key), `Config key "${key}" is missing from schema`).toBe(true);
    }
  });

  it('every schema field has a matching DOM control', () => {
    const container = document.createElement('div');
    container.innerHTML = CHUNK_HTML;
    document.body.appendChild(container);

    for (const field of schema.fields) {
      const control = container.querySelector(`[data-config-key="${field.key}"]`);
      expect(control, `No DOM control found for schema field "${field.key}"`).not.toBeNull();
    }

    document.body.removeChild(container);
  });

  it('no orphan DOM controls without schema entry', () => {
    const container = document.createElement('div');
    container.innerHTML = CHUNK_HTML;

    const configKeyEls = container.querySelectorAll('[data-config-key]');
    const schemaKeys = new Set(schema.fields.map(f => f.key));

    for (const el of configKeyEls) {
      const key = el.getAttribute('data-config-key');
      expect(schemaKeys.has(key), `Orphan DOM control for key "${key}" has no schema entry`).toBe(true);
    }
  });

  it('every schema field has help text', () => {
    for (const field of schema.fields) {
      expect(field.help, `Field "${field.key}" is missing help text`).toBeTruthy();
    }
  });

  it('read-only fields have a readOnlyReason', () => {
    for (const field of schema.fields) {
      if (field.readOnly) {
        expect(field.readOnlyReason, `Read-only field "${field.key}" is missing readOnlyReason`).toBeTruthy();
      }
    }
  });
});

describe('Round-trip write preservation', () => {
  it('unmodified write produces identical output', () => {
    if (!configText || !parsed) return;
    // Normalize line endings for comparison (parser normalizes \r\n → \n)
    const normalized = configText.replace(/\r\n/g, '\n');
    const output = writeConfig(configText, {}, schema.fields);
    expect(output).toBe(normalized);
  });

  it('modified value write preserves comments and unmodified lines', () => {
    if (!configText || !parsed) return;
    const patch = { CODE_LENGTH: 10 };
    const output = writeConfig(configText, patch, schema.fields);

    // The changed line should have the new value
    expect(output).toContain('CODE_LENGTH = 10');
    // Other lines should remain unchanged
    expect(output).toContain('MAX_TOKENS_FOR_NODE = 500');
    expect(output).toContain('import tiktoken');
    // Comments should be preserved
    expect(output).toContain('# The number of lines');
  });

  it('read-only keys are not written even if patched', () => {
    if (!configText || !parsed) return;
    const patch = { TOKENIZER: 'something_new', CODE_LENGTH: 7 };
    const output = writeConfig(configText, patch, schema.fields);

    // TOKENIZER should NOT be changed (read-only)
    expect(output).toContain('tiktoken.get_encoding');
    // CODE_LENGTH should be changed
    expect(output).toContain('CODE_LENGTH = 7');
  });

  it('boolean round-trip works correctly', () => {
    if (!configText || !parsed) return;
    const patch = { ENABLE_CODE_EXTRACTION: false };
    const output = writeConfig(configText, patch, schema.fields);
    expect(output).toContain('ENABLE_CODE_EXTRACTION = False');
  });

  it('string round-trip preserves quoted format', () => {
    if (!configText || !parsed) return;
    const patch = { OUTPUT_NAME: 'new_output.json' };
    const output = writeConfig(configText, patch, schema.fields);
    expect(output).toContain('OUTPUT_NAME = "new_output.json"');
  });
});
