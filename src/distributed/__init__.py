<<<<<<< HEAD

from . import ddp
from . import single
=======
from . import ddp, single
>>>>>>> otherfork/main

BACKEND_TYPE_TO_MODULE_MAP = {
    "nccl": ddp.DataParallelDistributedBackend,
    None: single.SinlgeNodeBackend,
}


def make_backend_from_args(args):
    return BACKEND_TYPE_TO_MODULE_MAP[args.distributed_backend](args)


def registered_backends():
    return BACKEND_TYPE_TO_MODULE_MAP.keys()
