from losses.kl_div_loss import KLDivLoss
from losses.mse_loss import MSELoss


LOSS_LUT = {
    "mse": MSELoss,
    "kl_div": KLDivLoss,
}
