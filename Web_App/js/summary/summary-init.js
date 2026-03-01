/**
 * summary-init.js — Summary tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initSummaryTab = createTabInit({
  phase: 'summary',
  schemaUrl: './config/summary.schema.json',
  configUrl: './Get_and_Chunk/config/summaryconfig.py',
  prefix: 'summary',
});
