import torch
from torch import nn
import torch.nn.functional as F

from lib import layers


class BaseNet(nn.Module):

    def __init__(self, nin, ch, nin_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, ch, 3, 1, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(ch * 6, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 8, dilations, dropout=True)
        self.lstm_aspp = layers.LSTMModule(ch * 8, nin_lstm)

        self.dec4 = layers.Decoder(ch * (6 + 8) + 1, ch * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 6), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(ch * 2, nin_lstm * 8)
        self.dec1 = layers.Decoder(ch * (1 + 2) + 1, ch * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)
        h = torch.cat([h, self.lstm_aspp(h)], dim=1)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft, predict_mag=True, predict_phase=True):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 16
        self.offset = 64
        self.predict_mag = predict_mag
        self.predict_phase = predict_phase

        self.stg1_low_band_net = BaseNet(2, 16, self.nin_lstm // 2)
        self.stg1_high_band_net = BaseNet(2, 16, self.nin_lstm // 2)

        self.stg2_bridge = layers.Conv2DBNActiv(18, 16, 1, 1, 0)
        self.stg2_low_band_net = BaseNet(16, 32, self.nin_lstm // 2)
        self.stg2_low_band_bridge = layers.Conv2DBNActiv(32, 16, 1, 1, 0)

        self.stg3_bridge = layers.Conv2DBNActiv(18, 16, 1, 1, 0)
        self.stg3_full_band_net = BaseNet(16, 32, self.nin_lstm)
        self.stg3_phase_net = BaseNet(20, 32, self.nin_lstm)

        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.phase_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(16, 2, 1, bias=False)

    def forward(self, x):
        x_mag = x[:, :2, :self.max_bin]
        x_phase = x[:, 2:, :self.max_bin]

        bandw = x_mag.size()[2] // 2
        l1_in = x_mag[:, :, :bandw]
        h1_in = x_mag[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)

        l2_in = torch.cat([l1_in, l1], dim=1)
        l2_in = self.stg2_bridge(l2_in)
        l2 = self.stg2_low_band_net(l2_in)
        aux = torch.cat([self.stg2_low_band_bridge(l2), h1], dim=2)

        mask = None
        if self.predict_mag:
            f3_in = torch.cat([x_mag, aux], dim=1)
            f3_in = self.stg3_bridge(f3_in)
            f3 = self.stg3_full_band_net(f3_in)

            mask = torch.sigmoid(self.out(f3))
            mask = F.pad(
                input=mask,
                pad=(0, 0, 0, self.output_bin - mask.size()[2]),
                mode='replicate'
            )

        phase_mask = None
        if self.predict_phase:
            p_in = torch.cat([x_mag, x_phase, aux], dim=1)
            p = self.stg3_phase_net(p_in)

            phase_mask = torch.sigmoid(self.phase_out(p))
            phase_mask = F.pad(
                input=phase_mask,
                pad=(0, 0, 0, self.output_bin - phase_mask.size()[2]),
                mode='replicate'
            )

        if self.training:
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode='replicate'
            )
            return mask, phase_mask, aux
        else:
            return mask, phase_mask

    def predict_mask(self, x):
        mask, phase_mask = self.forward(x)

        if self.predict_mag:
            if self.offset > 0:
                mask = mask[:, :, :, self.offset:-self.offset]
                assert mask.size()[3] > 0
        if self.predict_phase:
            if self.offset > 0:
                phase_mask = phase_mask[:, :, :, self.offset:-self.offset]
                assert phase_mask.size()[3] > 0

        return mask, phase_mask

    def predict(self, x):
        mask, phase_mask = self.forward(x)

        pred_mag = None
        if self.predict_mag:
            pred_mag = x[:, :2] * mask
            if self.offset > 0:
                pred_mag = pred_mag[:, :, :, self.offset:-self.offset]
                assert pred_mag.size()[3] > 0

        pred_phase = None
        if self.predict_phase:
            pred_phase = x[:, 2:] * phase_mask
            if self.offset > 0:
                pred_phase = pred_phase[:, :, :, self.offset:-self.offset]
                assert pred_phase.size()[3] > 0

        return pred_mag, pred_phase
