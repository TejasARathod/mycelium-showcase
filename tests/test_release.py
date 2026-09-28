"""Offline checks for copied source and reported summary integrity."""
import csv,hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReleaseChecks(unittest.TestCase):
 def test_source_provenance(self):
  for item in json.loads((ROOT/'docs/code-provenance.json').read_text())['files']:
   self.assertEqual(hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest(),item['sha256'],item['file'])
 def test_reported_counts(self):
  with (ROOT/'results/geographic_scores.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),4)
  for r in rows:
   for c in ['base','finetuned']:
    self.assertLessEqual(int(r[c+'_correct']),int(r[c+'_valid_n']))
  india=rows[0];self.assertAlmostEqual(int(india['base_correct'])/int(india['base_valid_n']),.716)
  self.assertAlmostEqual(int(india['finetuned_correct'])/int(india['finetuned_valid_n']),.984)
 def test_prompt_balance(self):
  spec=importlib.util.spec_from_file_location('prompts',ROOT/'src/mycelium/prompts.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  self.assertEqual(module.N_SEEDS,250);self.assertEqual(len(module.PROMPTS),4)
  for cell,phrases in module.PROMPTS.items():
   self.assertEqual(len(phrases),5)
   generated=[module.prompt_for_seed(cell,i)[1] for i in range(module.N_SEEDS)]
   for phrase in phrases:self.assertEqual(generated.count(phrase),50)
if __name__=='__main__':unittest.main()
