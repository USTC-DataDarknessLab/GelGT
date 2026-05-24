import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict
import wandb
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pynvml
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from model import GelGT
from utils import GloveTextEmbedding, GelGTTokens
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rel-f1')
parser.add_argument('--task', type=str, default='driver-top3')
parser.add_argument('--precompute', action='store_true', default=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--gt_conv_type', type=str, default='full')
parser.add_argument('--ablate', type=str, default='none')
parser.add_argument('--gnn_pe_dim', type=int, default=0)
parser.add_argument('--num_neighbors', type=int, default=800)
parser.add_argument('--sample_node_len', type=int, default=300)
parser.add_argument('--edge_dropout', type=float, default=0.2)
parser.add_argument('--num_centroids', type=int, default=4096)
parser.add_argument('--ff_dropout', type=float, default=0.1)
parser.add_argument('--attn_dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--temporal_strategy', type=str, default='uniform')
parser.add_argument('--pos_enc', type=str, default='none')
parser.add_argument('--max_degree', type=int, default=10000)
parser.add_argument('--pos_enc_dim', type=int, default=128)
parser.add_argument('--max_steps_per_epoch', type=int, default=3000)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='results/debug')
parser.add_argument('--run_name', type=str, default='debug')
parser.add_argument('--model_parameters', type=int, default=0, help='Number of model parameters')
parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('/back-up/lijin/.cache'))
parser.add_argument('--train_stage', type=str, default='finetune', choices=['finetune'])
args = parser.parse_args()
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
if local_rank == 0:
    args.run_name = f'{args.dataset}-{args.task}-{args.run_name}'

def init_gpu_utilization(device_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return handle

def get_gpu_stats(handle, device):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
    return (gpu_util, mem_allocated, mem_reserved)
print(f'Using device: {device}')
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)
gpu_handle = init_gpu_utilization(local_rank)
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)
stypes_cache_path = Path(f'{args.cache_dir}/{args.dataset}/stypes.json')
try:
    with open(stypes_cache_path, 'r') as f:
        col_to_stype_dict = json.load(f)
    for (table, col_to_stype) in col_to_stype_dict.items():
        for (col, stype_str) in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, 'w') as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)
(data, col_stats_dict) = make_pkey_fkey_graph(dataset.get_db(), col_to_stype_dict=col_to_stype_dict, text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=f'cuda:{local_rank}'), batch_size=256), cache_dir=f'{args.cache_dir}/{args.dataset}/materialized')
data = {split: GelGTTokens(data=data, task=task, K=args.num_neighbors, split=split, undirected=True, precompute=args.precompute, precomputed_dir=f'{args.cache_dir}/precomputed/{args.dataset}/{args.task}', num_workers=args.num_workers, train_stage=args.train_stage) for split in ['train', 'val', 'test']}
print('Scanning precomputed files for max relative time...')
max_time_list = []
for (split, dataset_obj) in data.items():
    t = dataset_obj.get_max_time()
    max_time_list.append(t)
GLOBAL_MAX_TIME = max(max_time_list)
if GLOBAL_MAX_TIME <= 0:
    GLOBAL_MAX_TIME = 365.0
if local_rank == 0:
    print(f'Detected Max Relative Time: {GLOBAL_MAX_TIME:.2f} days')
train_sampler = DistributedSampler(data['train'], shuffle=True, seed=args.seed)
loader_train = DataLoader(data['train'], batch_size=args.batch_size, sampler=train_sampler, collate_fn=data['train'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
val_sampler = DistributedSampler(data['val'], shuffle=False, seed=args.seed, drop_last=False)
loader_val = DataLoader(data['val'], batch_size=args.batch_size, sampler=val_sampler, collate_fn=data['val'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
test_sampler = DistributedSampler(data['test'], shuffle=False, seed=args.seed, drop_last=False)
loader_test = DataLoader(data['test'], batch_size=args.batch_size, sampler=test_sampler, collate_fn=data['test'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
loader_dict: Dict[str, DataLoader] = {'train': loader_train, 'val': loader_val, 'test': loader_test}
(clamp_min, clamp_max) = (None, None)
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = nn.BCEWithLogitsLoss()
    tune_metric = 'roc_auc'
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = 'mae'
    higher_is_better = False
    train_table = task.get_table('train')
    (clamp_min, clamp_max) = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = 'multilabel_auprc_macro'
    higher_is_better = True
else:
    raise ValueError(f'Task type {task.task_type} is unsupported')
model = GelGT(num_nodes=data['train'].data.num_nodes, max_neighbor_hop=data['train'].max_neighbor_hop, node_type_map=data['train'].node_type_to_index, col_names_dict={node_type: data['train'].data[node_type].tf.col_names_dict for node_type in data['train'].data.node_types}, col_stats_dict=col_stats_dict, local_num_layers=args.num_layers, channels=args.channels, out_channels=out_channels, global_dim=args.channels // 2, heads=args.num_heads, ff_dropout=args.ff_dropout, attn_dropout=args.attn_dropout, conv_type=args.gt_conv_type, ablate=args.ablate, gnn_pe_dim=args.gnn_pe_dim, num_centroids=args.num_centroids, sample_node_len=args.sample_node_len, max_time_days=GLOBAL_MAX_TIME, edge_dropout=args.edge_dropout, args=args).to(device)
for (name, param) in model.named_parameters():
    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)
for (name, buf) in model.named_buffers():
    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
if local_rank == 0:
    print(model)
total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
if local_rank == 0:
    print(f'Total model parameters: {total_params}')
args.model_parameters = total_params
if local_rank == 0:
    wandb.init(project='gel-gt-expts', name=args.run_name, config=vars(args))
output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)
world_size = dist.get_world_size()
base_lr = args.lr * world_size
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
global_step = 0
train_time = 0

def train_supervised(epoch) -> float:
    global global_step
    global train_time
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict['train']), args.max_steps_per_epoch)
    train_sampler.set_epoch(epoch)
    for (step, batch) in enumerate(tqdm(loader_dict['train'], total=total_steps, desc='Train'), start=1):
        neighbor_types = batch['neighbor_types'].to(device)
        node_indices = batch['node_indices'].to(device)
        neighbor_hops = batch['neighbor_hops'].to(device)
        neighbor_times = batch['neighbor_times'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_vec = batch['batch'].to(device)
        grouped_tf_dict = {'grouped_tfs': batch['grouped_tfs'], 'grouped_indices': batch['grouped_indices'], 'flat_batch_idx': batch['flat_batch_idx'], 'flat_nbr_idx': batch['flat_nbr_idx']}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        start_time = time.time()
        pred = model(neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred.float(), labels)
        loss.backward()
        train_time = train_time + time.time() - start_time
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_value = loss.detach().item()
        (gpu_util, mem_allocated, mem_reserved) = get_gpu_stats(gpu_handle, device)
        if local_rank == 0:
            wandb.log({'train_loss': loss_value, 'global_step': global_step, 'lr': optimizer.param_groups[0]['lr'], 'gpu_util_percent': gpu_util, 'gpu_mem_allocated_MB': mem_allocated, 'gpu_mem_reserved_MB': mem_reserved})
        loss_accum += loss_value * pred.size(0)
        count_accum += pred.size(0)
        global_step += 1
        if step >= args.max_steps_per_epoch:
            break
    return loss_accum / count_accum if count_accum > 0 else float('inf')
inference_time = 0

@torch.no_grad()
def test(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    global inference_time
    if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    eval_model.eval()
    pred_list = []
    idx_list = []
    for batch in tqdm(loader, desc=desc, disable=local_rank != 0):
        neighbor_types = batch['neighbor_types'].to(device)
        node_indices = batch['node_indices'].to(device)
        neighbor_hops = batch['neighbor_hops'].to(device)
        neighbor_times = batch['neighbor_times'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_vec = batch['batch'].to(device)
        grouped_tf_dict = {'grouped_tfs': batch['grouped_tfs'], 'grouped_indices': batch['grouped_indices'], 'flat_batch_idx': batch['flat_batch_idx'], 'flat_nbr_idx': batch['flat_nbr_idx']}
        start_time = time.time()
        pred = eval_model(neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        inference_time = time.time() - start_time
        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu().numpy())
        idx_list.append(batch['global_idx'].cpu().numpy())
    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs = np.concatenate(idx_list, axis=0) if idx_list else np.array([])
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)
    if local_rank == 0:
        all_preds = np.full((len(loader.dataset),), -100.0)
        for i in range(world_size):
            (g_idx, g_pred) = gathered[i]
            for (idx, pred) in zip(g_idx, g_pred):
                all_preds[idx] = pred
        return all_preds
    else:
        return None
if args.train_stage == 'finetune':
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    state_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_supervised(epoch)
        print(f'epoch:{epoch} Training Time is {train_time / epoch}')
        dist.barrier()
        eval_model = model.module
        val_pred = test(loader_dict['val'], eval_model=eval_model, epoch=epoch, desc='Val')
        test_pred = test(loader_dict['test'], eval_model=eval_model, epoch=epoch, desc='Test')
        if local_rank == 0:
            val_metrics = task.evaluate(val_pred, task.get_table('val'))
            test_metrics = task.evaluate(test_pred)
            print(f'Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics:{test_metrics}')
            wandb.log({'epoch': epoch, 'epoch_train_loss': train_loss, **{f'val_{k}': v for (k, v) in val_metrics.items()}})
           
            if higher_is_better and val_metrics[tune_metric] >= best_val_metric or (not higher_is_better and val_metrics[tune_metric] <= best_val_metric):
                best_val_metric = val_metrics[tune_metric]
                best_epoch = epoch
                state_dict = copy.deepcopy(model.module.state_dict())
                torch.save(state_dict, os.path.join(output_path, 'finetuned.pt'))
        dist.barrier()
    if local_rank == 0 and state_dict is not None:
        model.module.load_state_dict(state_dict)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()
    final_val_preds = test(loader_dict['val'], eval_model=model.module, epoch=best_epoch, desc='Val')
    final_test_preds = test(loader_dict['test'], eval_model=model.module, epoch=best_epoch, desc='Test')
    if local_rank == 0:
        val_metrics = task.evaluate(final_val_preds, task.get_table('val'))
        print(f'Best Val metrics: {val_metrics}')
        test_metrics = task.evaluate(final_test_preds)
        print(f'Best Test metrics: {test_metrics}')
        best_metrics_dict = {'val_metrics': val_metrics, 'test_metrics': test_metrics}
        file_path = os.path.join(output_path, str(args.seed) + '.json')
        with open(file_path, 'w') as f:
            json.dump(best_metrics_dict, f, indent=4)
    if local_rank == 0:
        print(f'[{args.train_stage.capitalize()} Stage] Training complete. No supervised evaluation performed.')
    print('train_time:', train_time)
    print('inference_time:', inference_time)
dist.destroy_process_group()