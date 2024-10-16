import argparse
import os
import sys

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from data.data import get_data
from models.model import *
from safetensors.torch import load_model
from torch_geometric.loader import DataLoader
from utils.logging_utils import Logger
from utils.metrics import *
from utils.utils import *


class PromptBindInference:
    def __init__(self, config_path='args.yml'):
        """
        Initializes the class with the given configuration path.

        Args:
            config_path (str): Path to the configuration file. Defaults to 'args.yml'.

        Attributes:
            config_path (str): Path to the configuration file.
            args (argparse.Namespace): Arguments loaded from the configuration file.
            accelerator: Accelerator setup for the model.
            logger: Logger setup for logging information.
            device (str): Device to be used for computation, set to 'cuda'.
            model: Loaded model.
            criterion: Main criterion for model evaluation.
            com_coord_criterion: Criterion for center of mass coordinates.
            pocket_cls_criterion: Criterion for pocket classification.
            pocket_coord_criterion: Criterion for pocket coordinates.
            test_loader: Data loader for test data.
            test_unseen_loader: Data loader for unseen test data.
        """
        self.config_path = config_path
        self.args = self.load_args()
        self.accelerator = self.setup_accelerator()
        self.logger = self.setup_logger()
        self.device = 'cuda'
        self.model = self.load_model()
        self.criterion, self.com_coord_criterion, self.pocket_cls_criterion, self.pocket_coord_criterion = self.setup_criterions()
        self.test_loader, self.test_unseen_loader = self.setup_data_loaders()

    def load_args(self):
        with open(self.config_path, 'r') as f:
            args_dict = yaml.safe_load(f)
        combined_args_dict = {**args_dict['config'], **args_dict['args']}

        prompt_nf = combined_args_dict.get('prompt_nf', '')
        combined_args_dict['pocket_prompt_nf'] = prompt_nf
        combined_args_dict['complex_prompt_nf'] = prompt_nf

        if 'exp_name' not in combined_args_dict:
            combined_args_dict['exp_name'] = f"test_prompt_{prompt_nf}"
        if 'ckpt' not in combined_args_dict:
            combined_args_dict['ckpt'] = f"pretrained/prompt_{prompt_nf}/best/model.safetensors"
            
        args = argparse.Namespace(**combined_args_dict)
        set_seed(args.seed)
        return args

    def setup_accelerator(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=self.args.mixed_precision)
        return accelerator

    def setup_logger(self):
        pre = f"{self.args.resultFolder}/{self.args.exp_name}"
        os.makedirs(pre, exist_ok=True)
        logger = Logger(accelerator=self.accelerator, log_path=f'{pre}/test.log')
        logger.log_message(f"{' '.join(sys.argv)}")
        return logger

    def load_model(self):
        model = get_model(self.args, self.logger, self.device)
        model = self.accelerator.prepare(model)
        load_model(model, self.args.ckpt)
        return model

    def setup_criterions(self):
        if self.args.pred_dis:
            criterion = nn.MSELoss()
            pred_dis = True
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.args.posweight))

        if self.args.coord_loss_function == 'MSE':
            com_coord_criterion = nn.MSELoss()
        elif self.args.coord_loss_function == 'SmoothL1':
            com_coord_criterion = nn.SmoothL1Loss()

        if self.args.pocket_cls_loss_func == 'bce':
            pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        pocket_coord_criterion = nn.HuberLoss(delta=self.args.pocket_coord_huber_delta)

        return criterion, com_coord_criterion, pocket_cls_criterion, pocket_coord_criterion

    def setup_data_loaders(self):
        if self.args.redocking:
            self.args.compound_coords_init_mode = "redocking"
        elif self.args.redocking_no_rotate:
            self.args.redocking = True
            self.args.compound_coords_init_mode = "redocking_no_rotate"

        train, valid, test = get_data(self.args, self.logger, addNoise=self.args.addNoise, use_whole_protein=self.args.use_whole_protein, compound_coords_init_mode=self.args.compound_coords_init_mode, pre=self.args.data_path)
        self.logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
        num_workers = 0

        test_loader = DataLoader(test, batch_size=self.args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
        test_unseen_pdb_list = [line.strip() for line in open('split_pdb_id/unseen_test_index')]

        test_unseen_index = test.data.query("(group =='test') and (pdb in @test_unseen_pdb_list)").index.values
        test_unseen_index_for_select = np.array([np.where(test._indices == i) for i in test_unseen_index]).reshape(-1)
        test_unseen = test.index_select(test_unseen_index_for_select)
        test_unseen_loader = DataLoader(test_unseen, batch_size=self.args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)

        return test_loader, test_unseen_loader

    def run_inference(self):
        """
        Runs the inference process for the model.

        This method sets the model to evaluation mode, logs the beginning of the test,
        and if the current process is the main process, it evaluates the model using
        the provided evaluation function. The evaluation metrics are then logged.

        The method waits for all processes to complete before finishing.

        Returns:
            None
        """
        self.model.eval()
        self.logger.log_message(f"Begin test")
        if self.accelerator.is_main_process:
            metrics, _, _ = evaluate_mean_pocket_cls_coord_multi_task(self.accelerator, self.args, self.test_unseen_loader, self.accelerator.unwrap_model(self.model), self.com_coord_criterion, self.criterion, self.pocket_cls_criterion, self.pocket_coord_criterion, self.args.relative_k,
                                                                      self.accelerator.device, pred_dis=self.args.pred_dis, use_y_mask=False, stage=2)
            self.logger.log_stats(metrics, 0, self.args, prefix="Test_unseen")
        self.accelerator.wait_for_everyone()

if __name__ == '__main__':
    infer = PromptBindInference(config_path='options/test_args.yml')
    infer.run_inference()