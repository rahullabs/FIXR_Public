# Based on https://github.com/aimagelab/mammoth
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from model.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='FIXR based on'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Fixr(ContinualModel):
    NAME = 'fixr'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args, transform):
        super(Fixr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
