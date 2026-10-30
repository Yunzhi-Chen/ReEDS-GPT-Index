"""Incremental index regression tests; uses synthetic embeddings, no API calls."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import build_reeds_index as builder

class Embedding(Embeddings):
    def __init__(self):
        self.embedded = []
    def embed_query(self, text):
        return [1., 0.]
    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [[1., 0.] for _ in texts]


def doc(text, path='model.gms', revision='current'):
    return Document(page_content=text, metadata={'rel_path':path,'file_sha1':revision})


class IncrementalIndexTests(unittest.TestCase):
    def setUp(self):
        self.embed=Embedding()
        self.pause=patch.object(builder,'EMBED_DELAY',0)
        self.pause.start()
    def tearDown(self):
        self.pause.stop()

    def test_repeated_update_is_idempotent_and_reuses_vectors(self):
        chunks=[doc('energy balance'),doc('power capacity'),doc('energy balance')]
        db=builder.replace_file(None,chunks,[],self.embed)
        ids=set(db.index_to_docstore_id.values())
        self.assertEqual(db.index.ntotal,2)
        self.assertEqual(len(self.embed.embedded),2)
        db=builder.replace_file(db,chunks,list(ids),self.embed)
        self.assertEqual(set(db.index_to_docstore_id.values()),ids)
        self.assertEqual(db.index.ntotal,2)
        self.assertEqual(len(self.embed.embedded),2)

    def test_changed_file_replaces_old_revision_and_embeds_only_new_text(self):
        db=builder.replace_file(None,[doc('retained'),doc('removed')],[],self.embed)
        old=list(db.index_to_docstore_id.values())
        db=builder.replace_file(db,[doc('retained',revision='new'),doc('added',revision='new')],old,self.embed)
        docs=[db.docstore.search(i) for i in db.index_to_docstore_id.values()]
        self.assertEqual({d.page_content for d in docs},{'retained','added'})
        self.assertEqual({d.metadata['file_sha1'] for d in docs},{'new'})
        self.assertEqual(self.embed.embedded,['retained','removed','added'])

    def test_stable_ids_preserve_identical_text_at_distinct_sources(self):
        a,b=doc('same','a.gms'),doc('same','b.gms')
        self.assertNotEqual(builder.stable_chunk_id(a),builder.stable_chunk_id(b))
        db=builder.replace_file(None,[a],[],self.embed)
        db=builder.replace_file(db,[b],[],self.embed)
        self.assertEqual(db.index.ntotal,2)

    def test_empty_file_removes_previous_chunks(self):
        db=builder.replace_file(None,[doc('removed')],[],self.embed)
        db=builder.replace_file(db,[],list(db.index_to_docstore_id.values()),self.embed)
        self.assertEqual(db.index.ntotal,0)

    def test_splitter_retains_original_character_offset(self):
        chunks=builder.make_splitter('.gms').split_documents([doc('eq_TEST(x) .. value =l= capacity;')])
        self.assertEqual(chunks[0].metadata['start_index'],0)

    def test_full_pipeline_replaces_changed_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';repo.mkdir();out=root/'index'
            (repo/'a.md').write_text('original energy explanation')
            (repo/'b.md').write_text('file later removed')
            with patch.dict(os.environ,{'REEDS_REPO':str(repo),'REEDS_INDEX_DIR':str(out)}), \
                 patch.object(builder,'make_embeddings',return_value=(self.embed,'test:model')):
                builder.main()
                (repo/'a.md').write_text('replacement energy explanation')
                (repo/'b.md').unlink()
                builder.main()
                db=FAISS.load_local(str(out),self.embed,allow_dangerous_deserialization=True)
                self.assertEqual(db.index.ntotal,1)
                self.assertEqual(db.docstore.search(db.index_to_docstore_id[0]).page_content,
                                 'replacement energy explanation')

    def test_load_failure_preserves_existing_files_instead_of_partial_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';repo.mkdir();out=root/'index'
            (repo/'a.md').write_text('valid source text')
            with patch.dict(os.environ,{'REEDS_REPO':str(repo),'REEDS_INDEX_DIR':str(out)}), \
                 patch.object(builder,'make_embeddings',return_value=(self.embed,'test:model')):
                builder.main()
                before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
                with patch.object(builder.FAISS,'load_local',side_effect=ValueError('test read failure')):
                    with self.assertRaisesRegex(RuntimeError,'preserve it and abort'):
                        builder.main()
                after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
                self.assertEqual(before,after)


if __name__=='__main__':
    unittest.main()
