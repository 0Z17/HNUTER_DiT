import unittest
import torch
from hnuter_dit.sdf_model import SDFPoseDiT
from hnuter_dit.moving_sdf_model import MovingSDFDiT
class MovingModelTests(unittest.TestCase):
 def test_transfer_and_train_only_normalizer_contract(self):
  base=SDFPoseDiT();model=MovingSDFDiT();model.initialize_from(base)
  for name in ('cnn','spatial','age','history'):
   self.assertTrue(all(not p.requires_grad for p in getattr(model,name).parameters()))
   for k,v in getattr(base,name).state_dict().items():torch.testing.assert_close(v,getattr(model,name).state_dict()[k])
  c=torch.zeros(2,17);c[:,3]=.5;x=torch.randn(2,8,6)
  torch.testing.assert_close(model.denormalized(model.normalized(x,c),c),x)
  self.assertEqual(tuple(model.baseline(c).shape),(2,8,6))
  self.assertTrue(model.blocks[0].parameters().__next__().requires_grad)
if __name__=='__main__':unittest.main()
