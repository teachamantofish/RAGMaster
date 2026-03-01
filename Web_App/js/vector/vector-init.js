/**
 * vector-init.js — Upsert to DB tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initVectorTab = createTabInit({
  phase: 'vector',
  schemaUrl: './config/vector.schema.json',
  configUrl: './VectorDB/config/vectorconfig.py',
  prefix: 'vector',
});
