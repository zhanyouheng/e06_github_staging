"""Verify the sealed delivery without inference, scoring, training, or writes."""
from common import *

def verify():
 assert verified(E/'DONE.json')
 m=readj(E/'manifest.json')
 assert sha(OUT/'REPORT_turn4.md')==m['report_sha256']
 count=0;size=0
 for r in rows(E/'artifacts.tsv'):
  p=OUT/r['path']
  assert p.is_relative_to(OUT) and not p.is_symlink(),str(p)
  assert p.stat().st_size==int(r['bytes']) and sha(p)==r['sha256'],str(p)
  count+=1;size+=p.stat().st_size
 assert count==m['payload_files'] and size==m['payload_bytes']
 checks=(E/'SHA256SUMS').read_text().splitlines()
 rr=list(rows(E/'artifacts.tsv'))
 assert checks==[r['sha256']+'  '+r['path'] for r in rr]
 assert readj(E/'revision/revision_evaluation_event.json')['overall_test_view_number']==2
 print(json.dumps({'passed':True,'payload_files':count,'payload_bytes':size,'test_reexecuted':False,'report_sha256':m['report_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':verify()
