import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'vector',
  schemaFile: 'vector.schema.json',
  configFile: 'vectorconfig.py',
  prefix: 'vector',
  configDir: 'VectorDB/config',
});
