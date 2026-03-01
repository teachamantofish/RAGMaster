/**
 * config-binder.test.js
 *
 * Unit tests for Web_App/js/config-binder.js
 * Uses jsdom to verify that schema fields are correctly bound to DOM controls.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { bindConfig } from '../../../js/config-binder.js';

// ---------- Helper: build a minimal chunk-like DOM fragment ----------
function createChunkDOM() {
  const container = document.createElement('div');
  container.id = 'tab-content-chunk';
  container.innerHTML = `
    <div class="config-field">
      <sl-input id="chunk-tokenizer" data-config-key="TOKENIZER" label="Tokenizer" disabled></sl-input>
    </div>
    <div class="config-field">
      <sl-input id="chunk-max_tokens_for_node" data-config-key="MAX_TOKENS_FOR_NODE" label="Max Tokens" type="number"></sl-input>
    </div>
    <div class="config-field">
      <sl-input id="chunk-chunk_size_range" data-config-key="CHUNK_SIZE_RANGE" label="Chunk Size Range" disabled></sl-input>
    </div>
    <div class="config-field">
      <sl-select id="chunk-chunk_model" data-config-key="CHUNK_MODEL" label="Chunk Model">
        <sl-option value="">Select a chunk model</sl-option>
        <sl-option value="heading-safe">heading-safe</sl-option>
        <sl-option value="greedy">greedy</sl-option>
        <sl-option value="fixed">fixed</sl-option>
      </sl-select>
    </div>
    <div class="config-field">
      <sl-input id="chunk-code_length" data-config-key="CODE_LENGTH" label="Code Length" type="number" min="1" max="50"></sl-input>
    </div>
    <div class="config-field">
      <sl-input id="chunk-keyword_density" data-config-key="KEYWORD_DENSITY" label="Keyword Density" type="number" step="0.01"></sl-input>
    </div>
    <div class="config-field">
      <sl-radio-group id="chunk-enable_code_extraction" data-config-key="ENABLE_CODE_EXTRACTION" label="Enable Code Extraction">
        <sl-radio value="true">True</sl-radio>
        <sl-radio value="false">False</sl-radio>
      </sl-radio-group>
    </div>
    <div class="config-field">
      <sl-input id="chunk-output_name" data-config-key="OUTPUT_NAME" label="Output File Name"></sl-input>
    </div>
  `;
  document.body.appendChild(container);
  return container;
}

// ---------- Test schema (minimal, matches chunk.schema.json) ----------
const testSchema = {
  fields: [
    { key: 'TOKENIZER', label: 'Tokenizer', type: 'text', readOnly: true, readOnlyReason: 'Expression', help: 'Tokenizer help' },
    { key: 'MAX_TOKENS_FOR_NODE', label: 'Max Tokens', type: 'number', min: 1, help: 'Max tokens help' },
    { key: 'CHUNK_SIZE_RANGE', label: 'Chunk Size Range', type: 'text', readOnly: true, readOnlyReason: 'Derived', help: 'Range help' },
    { key: 'CHUNK_MODEL', label: 'Chunk Model', type: 'select', options: [{ value: '', label: 'Select' }, { value: 'heading-safe', label: 'heading-safe' }, { value: 'greedy', label: 'greedy' }, { value: 'fixed', label: 'fixed' }], help: 'Model help' },
    { key: 'CODE_LENGTH', label: 'Code Length', type: 'number', min: 1, max: 50, help: 'Code length help' },
    { key: 'KEYWORD_DENSITY', label: 'Keyword Density', type: 'number', step: 0.01, min: 0, max: 1, help: 'Density help' },
    { key: 'ENABLE_CODE_EXTRACTION', label: 'Enable Code Extraction', type: 'boolean', help: 'Extraction help' },
    { key: 'OUTPUT_NAME', label: 'Output File Name', type: 'text', help: 'Output help' },
  ]
};

const testModel = {
  TOKENIZER: 'tiktoken.get_encoding("cl100k_base")',
  MAX_TOKENS_FOR_NODE: 500,
  CHUNK_SIZE_RANGE: 'f"max={MAX_TOKENS_FOR_NODE}"',
  CHUNK_MODEL: 'heading-safe',
  CODE_LENGTH: 3,
  KEYWORD_DENSITY: 0.2,
  ENABLE_CODE_EXTRACTION: true,
  OUTPUT_NAME: 'a_chunks.json',
};

describe('bindConfig', () => {
  let container;
  let helpCalls;
  let binder;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = createChunkDOM();
    helpCalls = [];
    binder = bindConfig(container, testSchema, { ...testModel }, (label, text) => {
      helpCalls.push({ label, text });
    });
  });

  it('populates text input with value', () => {
    const el = container.querySelector('[data-config-key="OUTPUT_NAME"]');
    expect(el.value).toBe('a_chunks.json');
  });

  it('populates number input with value', () => {
    const el = container.querySelector('[data-config-key="MAX_TOKENS_FOR_NODE"]');
    expect(el.value).toBe('500');
  });

  it('populates select with value', () => {
    const el = container.querySelector('[data-config-key="CHUNK_MODEL"]');
    expect(el.value).toBe('heading-safe');
  });

  it('populates boolean radio group with True', () => {
    const el = container.querySelector('[data-config-key="ENABLE_CODE_EXTRACTION"]');
    expect(el.value).toBe('true');
  });

  it('disables read-only fields', () => {
    const tokenizer = container.querySelector('[data-config-key="TOKENIZER"]');
    expect(tokenizer.disabled).toBe(true);

    const range = container.querySelector('[data-config-key="CHUNK_SIZE_RANGE"]');
    expect(range.disabled).toBe(true);
  });

  it('sets title on read-only fields', () => {
    const tokenizer = container.querySelector('[data-config-key="TOKENIZER"]');
    expect(tokenizer.title).toContain('Expression');
  });

  it('adds help icon for each field with help text', () => {
    const helpBtns = container.querySelectorAll('.config-help-btn');
    // Should have 8 help buttons (one per field)
    expect(helpBtns.length).toBe(8);
  });

  it('help icon click triggers onHelp callback', () => {
    const helpBtn = container.querySelector('.config-help-btn');
    helpBtn.click();
    expect(helpCalls.length).toBe(1);
    expect(helpCalls[0].label).toBeDefined();
    expect(helpCalls[0].text).toBeDefined();
  });

  it('collectValues returns only writable fields', () => {
    const collected = binder.collectValues();
    // TOKENIZER and CHUNK_SIZE_RANGE are read-only, should not be in result
    expect(collected).not.toHaveProperty('TOKENIZER');
    expect(collected).not.toHaveProperty('CHUNK_SIZE_RANGE');
    // Writable fields should be present
    expect(collected).toHaveProperty('MAX_TOKENS_FOR_NODE');
    expect(collected).toHaveProperty('CHUNK_MODEL');
    expect(collected).toHaveProperty('CODE_LENGTH');
    expect(collected).toHaveProperty('KEYWORD_DENSITY');
    expect(collected).toHaveProperty('ENABLE_CODE_EXTRACTION');
    expect(collected).toHaveProperty('OUTPUT_NAME');
  });

  it('validate passes with valid values', () => {
    const { valid, errors } = binder.validate();
    expect(valid).toBe(true);
    expect(errors.length).toBe(0);
  });

  it('validate fails when number out of range', () => {
    // Set CODE_LENGTH to 100 (max is 50)
    const el = container.querySelector('[data-config-key="CODE_LENGTH"]');
    el.value = '100';
    // Trigger change for model sync
    el.dispatchEvent(new Event('sl-change'));
    
    const { valid, errors } = binder.validate();
    expect(valid).toBe(false);
    expect(errors.some(e => e.key === 'CODE_LENGTH')).toBe(true);
  });
});
