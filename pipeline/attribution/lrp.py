import numpy as np
import torch
from zennit.attribution import Gradient
from zennit import composites as zcomp

from pipeline.base import BaseAttributor, BaseModel


_COMPOSITES = {
    "epsilon_plus_flat": zcomp.EpsilonPlusFlat,
    "epsilon_plus": zcomp.EpsilonPlus,
    "epsilon_alpha2_beta1_flat": zcomp.EpsilonAlpha2Beta1Flat,
    "epsilon_alpha2_beta1": zcomp.EpsilonAlpha2Beta1,
    "epsilon_gamma_box": zcomp.EpsilonGammaBox,
}


class LRPAttributor(BaseAttributor):
    """Layer-wise Relevance Propagation via zennit.

    Config keys:
        composite      : str    default "epsilon_plus_flat"
        target_class   : int    default 1
        batch_size     : int    default 256
        device         : str    default "cpu"
    """

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, model: BaseModel
    ) -> np.ndarray:
        composite_name = self.config.get("composite", "epsilon_plus_flat")
        target_class = int(self.config.get("target_class", 1))
        batch_size = int(self.config.get("batch_size", 256))
        device = self.config.get("device", "cpu")

        if composite_name not in _COMPOSITES:
            raise ValueError(
                f"Unknown zennit composite '{composite_name}'. "
                f"Available: {sorted(_COMPOSITES)}"
            )

        net = model.model
        if not isinstance(net, torch.nn.Module):
            raise TypeError(
                f"LRP requires a torch.nn.Module; got {type(net).__name__}"
            )
        net.eval().to(device)

        X_f = np.asarray(X, dtype=np.float32)
        n_samples = X_f.shape[0]

        with torch.no_grad():
            n_classes = net(torch.zeros(1, X_f.shape[1], device=device)).shape[-1]

        composite = _COMPOSITES[composite_name]()
        attributions = np.empty_like(X_f)

        with Gradient(model=net, composite=composite) as attributor:
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = torch.from_numpy(X_f[start:end]).to(device)

                target = torch.zeros(end - start, n_classes, device=device)
                target[:, target_class] = 1.0

                _, relevance = attributor(x_batch, target)
                attributions[start:end] = relevance.detach().cpu().numpy()

        assert attributions.shape == X_f.shape
        return attributions
