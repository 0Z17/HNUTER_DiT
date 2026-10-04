from collections import deque
from types import SimpleNamespace
import unittest
import numpy as np
from hnuter_dit.sdf_mapping import LocalSDF


class HistoryTests(unittest.TestCase):
    def mapper(self,times,last):
        mapper=LocalSDF.__new__(LocalSDF)
        mapper.history=deque((t,np.array([i])) for i,t in enumerate(times))
        mapper.last_time=last;mapper.latest_features=np.array([99])
        mapper.torch=SimpleNamespace(stack=np.stack,tensor=lambda v,**kw:np.asarray(v,dtype=np.float32),float32=np.float32)
        return mapper

    def test_latest_measurement_and_actual_ages(self):
        fields,ages=self.mapper([0,.201,.402,.603],.65).inputs()
        np.testing.assert_allclose(ages,[-.65,-.449,-.248,0],atol=1e-7)
        self.assertEqual(fields[-1,0],99)

    def test_unordered_or_stale_history_rejected(self):
        for times,last in (([0,.2,.4,.6],.3),([0,.2,.4,.6],1.2)):
            with self.assertRaises(ValueError):self.mapper(times,last).inputs()

    def test_missing_history_rejected(self):
        with self.assertRaises(ValueError):self.mapper([0,.2,.4],.4).inputs()


if __name__=='__main__':unittest.main()
