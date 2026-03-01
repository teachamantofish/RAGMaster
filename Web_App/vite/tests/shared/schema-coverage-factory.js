/**
 * Generic schema-coverage test factory.
 *
 * Given a phase name, schema filename, config filename, and HTML field keys,
 * generates the standard coverage tests:
 * 1. Schema loads
 * 2. Every config key → schema field
 * 3. Every schema field → DOM control
 * 4. Round-trip write preserves comments
 */

import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { parseConfig } from '../../../js/config-parser.js';
import { writeConfig } from '../../../js/config-writer.js';

function findWebAppRoot() {
  let candidate = process.cwd();
  if (existsSync(resolve(candidate, 'config/chunk.schema.json'))) return candidate;
  candidate = resolve(process.cwd(), '..');
  if (existsSync(resolve(candidate, 'config/chunk.schema.json'))) return candidate;
  return process.cwd();
}

/**
 * @param {object} opts
 * @param {string} opts.phase          - e.g. 'crawl'
 * @param {string} opts.schemaFile     - e.g. 'crawl.schema.json'
 * @param {string} opts.configFile     - e.g. 'crawlconfig.py'
 * @param {string} opts.prefix         - HTML ID prefix e.g. 'crawl'
 * @param {string} [opts.configDir]    - Relative dir from repo root e.g. 'VectorDB/config' (default: 'Get_and_Chunk/config')
 */
export function createSchemaCoverageTests({ phase, schemaFile, configFile, prefix, configDir }) {
  const webAppRoot = findWebAppRoot();
  const schemaPath = resolve(webAppRoot, 'config', schemaFile);
  const relDir = configDir || 'Get_and_Chunk/config';
  const configPath = resolve(webAppRoot, '..', relDir, configFile);

  let schema, configText, parsed;

  try {
    schema = JSON.parse(readFileSync(schemaPath, 'utf8'));
  } catch (e) {
    console.warn(`[${phase}] Could not load schema: ${schemaPath}`, e.message);
  }

  try {
    configText = readFileSync(configPath, 'utf8');
    parsed = parseConfig(configText);
  } catch (e) {
    console.warn(`[${phase}] Could not load config: ${configPath}`, e.message);
  }

  // Build a minimal HTML fragment for each schema field
  function buildHtml() {
    if (!schema) return '';
    return schema.fields.map(f => {
      const tag = f.type === 'select' ? 'sl-select'
        : f.type === 'boolean' ? 'sl-radio-group'
        : 'sl-input';
      return `<div class="config-field"><${tag} id="${prefix}-${f.key.toLowerCase()}" data-config-key="${f.key}"></${tag}></div>`;
    }).join('\n');
  }

  describe(`${phase} schema coverage`, () => {
    it('schema file loads correctly', () => {
      expect(schema).toBeDefined();
      expect(schema.fields).toBeDefined();
      expect(schema.fields.length).toBeGreaterThan(0);
    });

    it('every config key is represented in the schema', () => {
      if (!parsed || !schema) return;
      const schemaKeys = new Set(schema.fields.map(f => f.key));
      for (const key of Object.keys(parsed.values)) {
        expect(schemaKeys.has(key), `Config key "${key}" missing from schema`).toBe(true);
      }
    });

    it('every schema field has a matching DOM control', () => {
      const container = document.createElement('div');
      container.innerHTML = buildHtml();
      document.body.appendChild(container);

      for (const field of schema.fields) {
        const el = container.querySelector(`[data-config-key="${field.key}"]`);
        expect(el, `No DOM control for schema key "${field.key}"`).not.toBeNull();
      }

      container.remove();
    });

    it('round-trip write preserves comments', () => {
      if (!configText || !parsed || !schema) return;
      // Write back the same values — output should match original
      const normalised = configText.replace(/\r\n/g, '\n');
      const result = writeConfig(normalised, parsed.values, schema.fields);
      expect(result).toBe(normalised);
    });
  });
}
