import torch

from models import EATA


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {"EATA": EATA}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
            return device

        if (
            self.args.use_gpu
            and self.args.gpu_type == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            print("Use GPU: mps")
            return torch.device("mps")

        print("Use CPU")
        return torch.device("cpu")

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
