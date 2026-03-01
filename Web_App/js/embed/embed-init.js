/**
 * embed-init.js — Embed tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initEmbedTab = createTabInit({
  phase: 'embed',
  schemaUrl: './config/embed.schema.json',
  configUrl: './VectorDB/config/embedconfig.py',
  prefix: 'embed',
});
