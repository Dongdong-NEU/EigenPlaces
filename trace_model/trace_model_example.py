import torch
from pmodel.runner.tracer import BaseTracer
import torch.nn.functional  as F
H, W = 20, 20

class GridSample(torch.nn.Module):
    # due to the way of parser
    # which require model belong to `nn.Module`
    def __init__(self, H: int, W: int):
        super().__init__()
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        self._xx = xx.view(1, 1, H, W)
        self._yy = yy.view(1, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ run F.grid_sample for 4D tensor.

        Args:
            x (torch.Tensor): input and its format is NCHW

        Returns:
            out (torch.Tensor): the result of grid sample.
        """

        assert len(x.shape) == 4
        N = x.shape[0]
        grid = torch.cat(((self._xx.repeat(N, 1, 1, 1)), self._yy.repeat(N, 1, 1, 1)), 1) # N2HW
        grid = grid + 0.1
        grid = grid.permute(0, 2, 3, 1)  # N2HW -> NHW2
        out = F.grid_sample(x, grid, padding_mode="border")
        return out

if __name__ == "__main__":
    model = GridSample(H, W)
    x = torch.rand(1, 3, 20, 20)
    tracer = BaseTracer(model)
    tracer.trace(trace_data=(x,), trace_dir="./results", model_name="test_grid_sample")