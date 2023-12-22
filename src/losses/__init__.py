from losses.kl_div_loss import KLDivLoss
from losses.mse_loss import MSELoss
from losses.triplet_loss import TripletLoss


LOSS_LUT = {
    "mse": MSELoss,
    "kl_div": KLDivLoss,
    "triplet": TripletLoss,
}
