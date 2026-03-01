import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'crawl_pdf',
  schemaFile: 'crawl_pdf.schema.json',
  configFile: 'crawlpdfconfig.py',
  prefix: 'crawl-pdf',
});
