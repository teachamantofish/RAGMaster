import { createSchemaCoverageTests } from '../shared/schema-coverage-factory.js';

createSchemaCoverageTests({
  phase: 'embed',
  schemaFile: 'embed.schema.json',
  configFile: 'embedconfig.py',
  prefix: 'embed',
  configDir: 'VectorDB/config',
});
