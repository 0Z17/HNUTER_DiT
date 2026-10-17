import torch
from hnuter_dit.contact_inference import ddim_with_noise, latent_noise
from hnuter_dit.model import DiffusionSchedule, ddim_sample


def test_latents_are_independent_of_batch_order_and_model_scenario():
    whole = latent_noise(['a', 'b', 'a'], [0, 0, 1])
    torch.testing.assert_close(whole[1:], latent_noise(['b', 'a'], [0, 1]), rtol=0, atol=0)
    assert not torch.equal(whole[0], whole[2])


def test_explicit_noise_preserves_base_ddim_equations_and_endpoints():
    class Model(torch.nn.Module):
        def forward(self, x, t, c, o, mask):
            return x*.1 + c[:, :9, None].transpose(1, 2)*.01
    model = Model()
    c = torch.randn(2, 18)
    o = torch.zeros(2, 32, 10)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    schedule = DiffusionSchedule.cosine(100, torch.device('cpu'))
    noise = torch.randn((2, 128, 9), generator=torch.Generator().manual_seed(17))
    expected = ddim_sample(model, c, schedule, 128, 25, o, mask, torch.Generator().manual_seed(17))
    actual = ddim_with_noise(model, c, o, mask, None, noise, schedule)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[:, 0], c[:, :9])
    assert torch.equal(actual[:, -1], c[:, 9:])
