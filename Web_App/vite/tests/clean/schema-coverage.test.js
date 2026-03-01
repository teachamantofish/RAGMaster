import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'clean',
  schemaFile: 'clean.schema.json',
  configFile: 'cleanconfig.py',
  prefix: 'clean',
});
