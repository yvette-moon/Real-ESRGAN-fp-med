from collections import OrderedDict
import torch
import torch.nn.functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.srgan_model import SRGANModel
from realesrgan.archs.wavelet_ca_fuser_arch import denorm_subbands, haar_idwt2_torch

@MODEL_REGISTRY.register()
class WaveletMultiBranchModel(SRGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.gan_on_3ch = bool(opt.get("gan_on_3ch", True))
        self.percep_on_3ch = bool(opt.get("percep_on_3ch", True))

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def _reconstruct(self, bands):
        bands = torch.clamp(bands, 0.0, 1.0)
        ll, lh, hl, hh = torch.chunk(bands, 4, dim=1)
        ll, lh, hl, hh = denorm_subbands(ll, lh, hl, hh)
        recon = haar_idwt2_torch(ll, lh, hl, hh)
        return torch.clamp(recon, 0.0, 1.0)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        pred_bands = self.net_g(self.lq)
        self.output = self._reconstruct(pred_bands)

        l_g_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict["l_g_pix"] = l_g_pix

        if self.cri_perceptual:
            out_p = self.output.repeat(1, 3, 1, 1) if self.percep_on_3ch else self.output
            gt_p = self.gt.repeat(1, 3, 1, 1) if self.percep_on_3ch else self.gt
            l_g_percep, l_g_style = self.cri_perceptual(out_p, gt_p)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict["l_g_percep"] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict["l_g_style"] = l_g_style

        if self.cri_gan:
            for p in self.net_d.parameters():
                p.requires_grad = False
            gan_in = self.output.repeat(1, 3, 1, 1) if self.gan_on_3ch else self.output
            fake_g_pred = self.net_d(gan_in)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict["l_g_gan"] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        if self.cri_gan:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            real_in = self.gt.repeat(1, 3, 1, 1) if self.gan_on_3ch else self.gt
            real_d_pred = self.net_d(real_in)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict["l_d_real"] = l_d_real
            l_d_real.backward()

            fake_in = self.output.detach()
            fake_in = fake_in.repeat(1, 3, 1, 1) if self.gan_on_3ch else fake_in
            fake_d_pred = self.net_d(fake_in)
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict["l_d_fake"] = l_d_fake
            l_d_fake.backward()

            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            pred_bands = self.net_g(self.lq)
            self.output = self._reconstruct(pred_bands)
        self.net_g.train()