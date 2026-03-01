/**
 * clean-init.js — Clean tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initCleanTab = createTabInit({
  phase: 'clean',
  schemaUrl: './config/clean.schema.json',
  configUrl: './Get_and_Chunk/config/cleanconfig.py',
  prefix: 'clean',
});
