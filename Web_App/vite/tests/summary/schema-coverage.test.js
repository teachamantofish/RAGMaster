import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'summary',
  schemaFile: 'summary.schema.json',
  configFile: 'summaryconfig.py',
  prefix: 'summary',
});
