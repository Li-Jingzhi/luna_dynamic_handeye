import json
from pathlib import Path
import tempfile
import unittest
from data_collection.saved_preview_stream import tail_record

class SavedPreviewTests(unittest.TestCase):
    def test_tail_reads_last_complete_record_without_loading_full_log(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'frames.jsonl'
            p.write_bytes((b'{"old":0}\n'*100000)+json.dumps(dict(id='latest',measurement='valid')).encode()+b'\n{"partial":')
            self.assertEqual(tail_record(p,1024)['id'],'latest')
            p.write_text('{"partial":')
            with self.assertRaises(ValueError):tail_record(p)
