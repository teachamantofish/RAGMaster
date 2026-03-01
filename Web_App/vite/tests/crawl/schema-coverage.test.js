import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'crawl',
  schemaFile: 'crawl.schema.json',
  configFile: 'crawlconfig.py',
  prefix: 'crawl',
});
