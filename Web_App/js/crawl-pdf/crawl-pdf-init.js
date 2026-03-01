/**
 * crawl-pdf-init.js — Crawl PDF tab lifecycle controller.
 */
import { createTabInit } from '../tab-init.js';

export const initCrawlPdfTab = createTabInit({
  phase: 'crawl_pdf',
  schemaUrl: './config/crawl_pdf.schema.json',
  configUrl: './Get_and_Chunk/config/crawlpdfconfig.py',
  prefix: 'crawl-pdf',
});
