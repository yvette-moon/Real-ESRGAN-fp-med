from collections import OrderedDict
import torch
import torch.nn.functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel


@MODEL_REGISTRY.register()
class WaveletRealESRGANModel(RealESRGANModel):
    """RealESRGAN + Haar wavelet sub-band GAN constraints."""

    def __init__(self, opt):
        super(WaveletRealESRGANModel, self).__init__(opt)
        self.wavelet_gan_weight = float(opt.get("wavelet_gan_weight", 0.05))
        self.wavelet_pixel_weight = float(opt.get("wavelet_pixel_weight", 0.0))
        self.wavelet_use_ll = bool(opt.get("wavelet_use_ll", True))
        self.wavelet_use_hf = bool(opt.get("wavelet_use_hf", True))
        self.wavelet_loss_on_usm = bool(opt.get("wavelet_loss_on_usm", True))

    def _active_bands(self):
        bands = []
        if self.wavelet_use_ll:
            bands.append("ll")
        if self.wavelet_use_hf:
            bands.extend(["lh", "hl", "hh"])
        return bands

    def _haar_subbands_for_gan(self, x):
        # Use first channel for wavelet analysis (CT pipeline already replicated 3 channels).
        if x.shape[1] > 1:
            x = x[:, :1, :, :]

        h, w = x.shape[-2], x.shape[-1]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        weight = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],      # LL
                [[-0.5, -0.5], [0.5, 0.5]],    # LH
                [[-0.5, 0.5], [-0.5, 0.5]],    # HL
                [[0.5, -0.5], [-0.5, 0.5]],    # HH
            ],
            dtype=x.dtype,
            device=x.device
        ).unsqueeze(1)  # [4, 1, 2, 2]

        y = F.conv2d(x, weight, stride=2)
        ll, lh, hl, hh = torch.chunk(y, 4, dim=1)

        # Map to [0,1] so discriminator sees similar range as image domain.
        ll = torch.clamp(ll / 2.0, 0.0, 1.0)
        lh = torch.clamp((lh + 1.0) / 2.0, 0.0, 1.0)
        hl = torch.clamp((hl + 1.0) / 2.0, 0.0, 1.0)
        hh = torch.clamp((hh + 1.0) / 2.0, 0.0, 1.0)

        return {
            "ll": ll.repeat(1, 3, 1, 1),
            "lh": lh.repeat(1, 3, 1, 1),
            "hl": hl.repeat(1, 3, 1, 1),
            "hh": hh.repeat(1, 3, 1, 1),
        }

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm if self.opt["l1_gt_usm"] else self.gt
        percep_gt = self.gt_usm if self.opt["percep_gt_usm"] else self.gt
        gan_gt = self.gt_usm if self.opt["gan_gt_usm"] else self.gt

        # ---------------- optimize G ----------------
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        fake_sub = self._haar_subbands_for_gan(self.output)
        real_for_wavelet = gan_gt if self.wavelet_loss_on_usm else self.gt
        real_sub = self._haar_subbands_for_gan(real_for_wavelet)

        l_g_total = 0
        loss_dict = OrderedDict()

        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict["l_g_pix"] = l_g_pix

            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict["l_g_percep"] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict["l_g_style"] = l_g_style

            # image-domain GAN
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict["l_g_gan"] = l_g_gan

            # wavelet sub-band GAN
            for b in self._active_bands():
                pred_b = self.net_d(fake_sub[b])
                l_b_gan = self.cri_gan(pred_b, True, is_disc=False) * self.wavelet_gan_weight
                l_g_total += l_b_gan
                loss_dict[f"l_g_gan_{b}"] = l_b_gan

                if self.wavelet_pixel_weight > 0:
                    l_b_pix = F.l1_loss(fake_sub[b], real_sub[b]) * self.wavelet_pixel_weight
                    l_g_total += l_b_pix
                    loss_dict[f"l_g_pix_{b}"] = l_b_pix

            l_g_total.backward()
            self.optimizer_g.step()

        # ---------------- optimize D ----------------
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # image-domain D
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict["l_d_real"] = l_d_real
        loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(self.output.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict["l_d_fake"] = l_d_fake
        loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()

        # wavelet-domain D
        for b in self._active_bands():
            real_b_pred = self.net_d(real_sub[b])
            l_d_real_b = self.cri_gan(real_b_pred, True, is_disc=True) * self.wavelet_gan_weight
            loss_dict[f"l_d_real_{b}"] = l_d_real_b
            l_d_real_b.backward()

            fake_b_pred = self.net_d(fake_sub[b].detach())
            l_d_fake_b = self.cri_gan(fake_b_pred, False, is_disc=True) * self.wavelet_gan_weight
            loss_dict[f"l_d_fake_{b}"] = l_d_fake_b
            l_d_fake_b.backward()

        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
