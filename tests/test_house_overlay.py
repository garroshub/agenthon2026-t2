from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import t2_forecaster.house_overlay as house

class HouseOverlayTests(unittest.TestCase):
    def _text_dir(self,text,doc_type="fomc_statement"):
        td=tempfile.TemporaryDirectory(); root=Path(td.name)
        (root/"policy.txt").write_text(text,encoding="utf-8")
        (root/"corpus_index.json").write_text(json.dumps({"documents":[{"doc_id":"policy-doc","file":"policy.txt","doc_type":doc_type,"timestamp":"2024-01-31T19:00:00Z"}]}),encoding="utf-8")
        return td,root
    def _claim(self,text,direction,fact_kind="conditional_or_forward_guidance",actor="committee",temporal="future",confidence=.95,conditional=True,negated=False):
        return {"claims":[{"fact_kind":fact_kind,"actor_scope":actor,"temporal_scope":temporal,"direction":direction,"conditional":conditional,"negated":negated,"confidence":confidence,"span_start":0,"span_end":len(text),"quote":text}]}
    def test_valid_tightening_shifts_center_only(self):
        text="The Committee expects further policy firming may be appropriate."; td,root=self._text_dir(text)
        try:
            rng=np.random.default_rng(7); draws=rng.normal(size=(1000,1,2))
            with patch.object(house,"_house",return_value=self._claim(text,"tightening")):
                got=house.maybe_apply(draws,target_type="level",target_frequency="daily",assets=("UST_2Y",),asof="2024-01-31",text_dir=root)
            self.assertTrue(got.applied)
            for j in range(2):
                expected=house.SHIFT*.95*np.std(draws[:,0,j],ddof=1)
                np.testing.assert_allclose(got.draws[:,0,j]-draws[:,0,j],expected)
                self.assertAlmostEqual(float(np.std(got.draws[:,0,j],ddof=1)),float(np.std(draws[:,0,j],ddof=1)),places=12)
        finally: td.cleanup()
    def test_valid_easing_shifts_down(self):
        text="The Committee expects reducing policy restraint will be appropriate."; td,root=self._text_dir(text)
        try:
            draws=np.arange(1000,dtype=float).reshape(1000,1,1)/100.
            with patch.object(house,"_house",return_value=self._claim(text,"easing")):
                got=house.maybe_apply(draws,target_type="level",target_frequency="daily",assets=("UST_10Y",),asof="2024-01-31",text_dir=root)
            self.assertTrue(got.applied); self.assertLess(float(np.mean(got.draws-draws)),0.)
        finally: td.cleanup()
    def test_announced_decision_is_noop(self):
        text="The Committee decided to raise the target range today."; td,root=self._text_dir(text)
        try:
            draws=np.ones((1000,1,1)); parsed=self._claim(text,"tightening",fact_kind="announced_decision",temporal="current")
            with patch.object(house,"_house",return_value=parsed):
                got=house.maybe_apply(draws,target_type="level",target_frequency="daily",assets=("UST_2Y",),asof="2024-01-31",text_dir=root)
            self.assertFalse(got.applied); np.testing.assert_array_equal(got.draws,draws)
        finally: td.cleanup()
    def test_conflicting_guidance_is_noop(self):
        text="Further firming may be appropriate. Reducing restraint may also become appropriate."; td,root=self._text_dir(text)
        try:
            split=text.index(".")+1
            parsed={"claims":[
                {"fact_kind":"conditional_or_forward_guidance","actor_scope":"committee","temporal_scope":"future","direction":"tightening","conditional":True,"negated":False,"confidence":.95,"span_start":0,"span_end":split,"quote":text[:split]},
                {"fact_kind":"conditional_or_forward_guidance","actor_scope":"committee","temporal_scope":"future","direction":"easing","conditional":True,"negated":False,"confidence":.95,"span_start":split+1,"span_end":len(text),"quote":text[split+1:]}]}
            draws=np.arange(1000,dtype=float).reshape(1000,1,1)
            with patch.object(house,"_house",return_value=parsed):
                got=house.maybe_apply(draws,target_type="level",target_frequency="daily",assets=("UST_2Y",),asof="2024-01-31",text_dir=root)
            self.assertFalse(got.applied); np.testing.assert_array_equal(got.draws,draws)
        finally: td.cleanup()
    def test_ineligible_contract_never_calls_house(self):
        draws=np.ones((1000,2,1))
        with patch.object(house,"_house",side_effect=AssertionError("must not call")):
            got=house.maybe_apply(draws,target_type="log_return",target_frequency="daily",assets=("MOM","HML"),asof="2024-01-31",text_dir="missing")
        self.assertFalse(got.applied); np.testing.assert_array_equal(got.draws,draws)
if __name__=="__main__": unittest.main()
