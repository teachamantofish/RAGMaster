/**
 * crawl-init.js — Crawl Web tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initCrawlTab = createTabInit({
  phase: 'crawl',
  schemaUrl: './config/crawl.schema.json',
  configUrl: './Get_and_Chunk/config/crawlconfig.py',
  prefix: 'crawl',
});
