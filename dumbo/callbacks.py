from schedulefree.adamw_schedulefree import AdamWScheduleFree
from schedulefree.sgd_schedulefree import SGDScheduleFree
from composer import Callback, Logger, State

class ScheduleFreeCallback(Callback):
    def batch_start(self, state: State, _: Logger) -> None:
        for optim in state.optimizers:
            if isinstance(optim, AdamWScheduleFree) or isinstance(optim, SGDScheduleFree):
                optim.train()
    
    def eval_before_all(self, state: State, _: Logger) -> None:
        for optim in state.optimizers:
            if isinstance(optim, AdamWScheduleFree) or isinstance(optim, SGDScheduleFree):
                optim.eval()

class ExtraLoggingCallback(Callback):
    def batch_end(self, state: State, logger: Logger) -> None:
        #logger.log_metrics(state.batch_metrics, step=state.global_step)
        for idx, scheduler in enumerate(state.schedulers):
            logger.log_metrics(
                {
                    f"scheduler/{idx}/lr": scheduler.get_last_lr()[0]
                }
            )
