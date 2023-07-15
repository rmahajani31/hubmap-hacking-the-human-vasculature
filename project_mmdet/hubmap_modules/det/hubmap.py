import os
import shutil
import json

from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class ModelCheckpointingHook(Hook):
    def __init__(self, interval, metrics_file_name, chkp_dir, chkp_name):
        self.interval = interval
        self.chkp_dir = chkp_dir
        self.chkp_name = chkp_name
        self.metrics_file_path = f'{self.chkp_dir}/{metrics_file_name}'
        self.max_ap = 0
    
    def before_train(self, runner):
        if os.path.exists(self.chkp_dir):
            shutil.rmtree(self.chkp_dir)
        os.mkdir(self.chkp_dir)
        if os.path.exists(self.metrics_file_path):
            os.remove(self.metrics_file_path)
    
    def after_val_epoch(self, runner, metrics):
        if self.every_n_epochs(runner, self.interval):
            fp = open(self.metrics_file_path, 'a')
            fp.write(f'Epoch {runner.epoch}.....\n')
            json.dump(metrics, fp)
            fp.write(f'\n===============\n')
            fp.close()
            if 'coco/bbox_mAP' in metrics:
                cur_ap = metrics['coco/bbox_mAP']
                meta = dict(epoch=runner.epoch, iter=runner.iter)
                self.chkp_name_parts = self.chkp_name.split('.pth')
                if cur_ap > self.max_ap:
                    chkp_pth_files = [x for x in os.listdir(self.chkp_dir) if '.pth' in x]
                    for chkp_pth_file in chkp_pth_files:
                        os.remove(f'{self.chkp_dir}/{chkp_pth_file}')
                    runner.save_checkpoint(
                        self.chkp_dir,
                        filename=f'{self.chkp_name_parts[0]}_{cur_ap}.pth',
                        file_client_args=None,
                        save_optimizer=False,
                        save_param_scheduler=False,
                        meta=meta,
                        by_epoch=False,
                        backend_args=None)
                    self.max_ap = cur_ap