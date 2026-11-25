import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import utils.data_loaders
import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.defect_metrics import get_defect_boundary_cd
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils_clamp import get_loss_clamp, get_loss_mvp
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from utils.mvp_utils import *
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # self.lr_decays = {i: 0.1 ** (1 / cfg.TRAIN.LR_DECAY) for i in range(1, cfg.TRAIN.N_EPOCHS+1)}
        # self.scheduler_steplr = StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=self.scheduler_steplr)
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0

        # Training history: (epoch, cdc, cd1, cd2, cd3, partial_matching, f_score).
        self.history = []
        self.cfg = cfg

    def _should_save_heatmap(self, record):
        if not record:
            return False
        if not getattr(self.cfg.TRAIN, 'HEATMAP_ENABLED', False):
            return False
        interval = max(1, int(getattr(self.cfg.TRAIN, 'HEATMAP_SAVE_EVERY', 10)))
        return self.epoch == 1 or self.epoch % interval == 0

    def _compute_pred_error_colors(self, pred_points, gt_points):
        pred_tensor = torch.from_numpy(pred_points).float().unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_points).float().unsqueeze(0)
        distances = torch.cdist(pred_tensor, gt_tensor, p=2).squeeze(0)
        errors = distances.min(dim=1).values.cpu().numpy()
        vmax = np.percentile(errors, 95)
        vmax = max(vmax, 1e-8)
        normalized = np.clip(errors / vmax, 0.0, 1.0)
        colors = plt.get_cmap('turbo')(normalized)[:, :3]
        return colors, errors

    def _save_training_heatmap(self, sample):
        out_dir = os.path.join(self.cfg.DIR.LOGS, 'heatmaps')
        os.makedirs(out_dir, exist_ok=True)

        partial = sample['partial']
        gt = sample['gt']
        pred = sample['pred']
        colors, errors = self._compute_pred_error_colors(pred, gt)

        img_gt = pc_util.point_cloud_three_views(gt, diameter=7)
        img_pred = pc_util.point_cloud_three_views(pred, diameter=7)
        img_heat = pc_util.point_cloud_three_views_with_colors(pred, colors, diameter=7)

        comp = np.concatenate([img_gt, img_pred, img_heat], axis=0)
        comp = (comp * 255).astype('uint8')
        im = Image.fromarray(comp)

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(im)
        h = img_gt.shape[0]
        try:
            font = ImageFont.truetype("Times New Roman.ttf", size=32)
        except Exception:
            try:
                font = ImageFont.truetype("Times.ttf", size=32)
            except Exception:
                font = ImageFont.load_default()

        draw.text((40, 30), 'GT', fill=(0, 0, 0), font=font)
        draw.text((40, h + 30), 'Prediction', fill=(0, 0, 0), font=font)
        draw.text((40, 2 * h + 30), 'Error Heatmap', fill=(0, 0, 0), font=font)

        stats_text = (
            f"Epoch {self.epoch} | Sample {sample['model_id']} | "
            f"mean={errors.mean():.4f} max={errors.max():.4f}"
        )
        draw.text((40, comp.shape[0] - 50), stats_text, fill=(0, 0, 0), font=font)

        save_path = os.path.join(out_dir, f'epoch_{self.epoch:03d}_{sample["model_id"]}_heatmap.png')
        im.save(save_path)
        print('Training heatmap saved to: {}'.format(save_path))

    def _restore_best_checkpoint(self, model, cfg, reason):
        best_ckpt = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
        if not os.path.exists(best_ckpt):
            print('\n[Error] {} No checkpoint is available; training stops.'.format(reason))
            return False

        print('\n[Warning] {} Restoring from best checkpoint (Epoch {}).'.format(reason, self.best_epoch))
        device = next(model.parameters()).device
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        return True


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | f_score | cd_defect | cd_boundary | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            
            # Show training progress with tqdm.
            pbar = tqdm(enumerate(train_data_loader), total=n_batches, 
                       desc='Epoch {:3d}/{:3d}'.format(epoch_idx, cfg.TRAIN.N_EPOCHS), 
                       ncols=120, leave=True)
            skip_validation = False
            for batch_idx, (taxonomy_ids, model_ids, data) in pbar:
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = data['partial_cloud']
                gt = data['gtcloud']

                pcds_pred = model(partial)

                # Select CDL1/CDL2 according to cfg.TEST.CD_TYPE.
                use_sqrt = getattr(cfg.TEST, 'CD_TYPE', 'L2') == 'L2'
                loss_total, losses, gts = get_loss_clamp(
                    pcds_pred, partial, gt, sqrt=use_sqrt,
                    fscore_weight=getattr(cfg.TRAIN, 'FSCORE_LOSS_WEIGHT', 0.0),
                    fscore_threshold=getattr(cfg.TRAIN, 'FSCORE_LOSS_THRESHOLD', 0.01),
                    fscore_softness=getattr(cfg.TRAIN, 'FSCORE_LOSS_SOFTNESS', 0.002)
                )

                if not torch.isfinite(loss_total):
                    skip_validation = True
                    if not self._restore_best_checkpoint(model, cfg, 'Non-finite training loss detected; rollback was applied.'):
                        break
                    model.train()
                    break

                self.optimizer.zero_grad()
                loss_total.backward()
                # Clip gradients to reduce the risk of exploding gradients and NaNs.
                if hasattr(cfg.TRAIN, 'GRAD_CLIP') and cfg.TRAIN.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # Update the progress bar with the current loss, including NaN formatting.
                cd3_str = '{:.4f}'.format(cd_p3_item) if not np.isnan(cd_p3_item) else 'nan'
                loss_str = '{:.4f}'.format(loss_total.item()) if not np.isnan(loss_total.item()) else 'nan'
                pbar.set_postfix({'cd3': cd3_str, 'loss': loss_str})

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

            if skip_validation:
                continue

            # Update learning rate after optimizer.step() (PyTorch 1.1+ requirement)
            self.lr_scheduler.step()

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # Training record with NaN-safe formatting.
            loss_strs = ['%.4f' % l if not np.isnan(l) else 'nan' for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, loss_strs))

            # Validate the current model
            cd_eval, metrics_dict, val_message = self.validate(
                cfg, model=model, val_data_loader=val_data_loader, record=False
            )
            metrics_dict['epoch'] = epoch_idx
            self.history.append(metrics_dict)

            # Print validation metrics for the current epoch, including F-score.
            f_score_val = metrics_dict['f_score']
            cd3_str = '{:.4f}'.format(cd_eval) if not np.isnan(cd_eval) else 'nan'
            f_str = '{:.4f}'.format(f_score_val) if not np.isnan(f_score_val) else 'nan'
            print(
                '  [Val] Epoch {:3d}: CD3 = {}, F-score = {}, CD-Defect = {:.4f}, CD-boundary = {:.4f}'.format(
                    epoch_idx, cd3_str, f_str, metrics_dict['cd_defect'], metrics_dict['cd_boundary']
                )
            )

            previous_history = self.history[:-1]
            previous_best_f = max(
                [h['f_score'] for h in previous_history if not np.isnan(h['f_score'])],
                default=0.0
            )
            anomaly_reason = None
            if np.isnan(cd_eval) or np.isnan(avg_cdc):
                anomaly_reason = 'NaN detected; rollback was applied.'
            elif self.best_epoch > 0 and epoch_idx >= getattr(cfg.TRAIN, 'ANOMALY_MIN_EPOCHS', 10):
                cd_factor = getattr(cfg.TRAIN, 'ANOMALY_CD_FACTOR', 3.0)
                f_ratio = getattr(cfg.TRAIN, 'ANOMALY_F_SCORE_RATIO', 0.25)
                f_cd_factor = getattr(cfg.TRAIN, 'ANOMALY_F_SCORE_CD_FACTOR', 1.5)
                if np.isfinite(cd_eval) and cd_eval > self.best_metrics * cd_factor:
                    anomaly_reason = 'Validation CD3 spike detected ({:.4f} > {:.2f} x {:.4f}); rollback was applied.'.format(
                        cd_eval, cd_factor, self.best_metrics)
                elif previous_best_f > 0 and f_score_val < previous_best_f * f_ratio:
                    if np.isfinite(cd_eval) and cd_eval > self.best_metrics * f_cd_factor:
                        anomaly_reason = (
                            'Validation F-score drop with simultaneous CD3 degradation detected '
                            '({:.4f} < {:.2f} x {:.4f}, {:.4f} > {:.2f} x {:.4f}); rollback was applied.'
                        ).format(
                            f_score_val, f_ratio, previous_best_f,
                            cd_eval, f_cd_factor, self.best_metrics)
                    else:
                        print(
                            '  [Warn] F-score dropped temporarily without CD3 degradation; skipping rollback '
                            '({:.4f} < {:.2f} x {:.4f}, CD3={:.4f}, best={:.4f})'.format(
                                f_score_val, f_ratio, previous_best_f,
                                cd_eval, self.best_metrics)
                        )

            if anomaly_reason is not None:
                self.history.pop()
                if not self._restore_best_checkpoint(model, cfg, anomaly_reason):
                    break
                continue

            self.test_record(val_message)

            # Save checkpoints
            if cd_eval < self.best_metrics:
                self.best_epoch = epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }, output_path)

                print('Saved checkpoint to %s ...' % output_path)
                self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()

        # After training: summarize metrics, plot curves, save Excel, and report per-category F-score.
        self._print_training_summary()
        self._plot_metrics_curves()
        self._save_metrics_excel()
        self._evaluate_per_category_fscore(model, val_data_loader, cfg)

    def validate(self, cfg, model=None, val_data_loader=None, outdir=None, compute_f_score=True, record=True):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        defect_metrics = AverageMeter(['cd_defect', 'cd_boundary'])
        f_score_sum = 0.0
        f_score_count = 0
        save_heatmap = self._should_save_heatmap(record)
        heatmap_sample = None
        target_sample_index = int(getattr(cfg.TRAIN, 'HEATMAP_SAMPLE_INDEX', 0))
        seen_samples = 0

        # Start testing with progress bar
        pbar = tqdm(enumerate(val_data_loader), total=n_samples, desc='Validating', ncols=100, leave=False)
        for model_idx, (taxonomy_id, model_id, data) in pbar:
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = data['partial_cloud']
                gt = data['gtcloud']

                # forward
                pcds_pred = model(partial.contiguous())
                use_sqrt = getattr(cfg.TEST, 'CD_TYPE', 'L2') == 'L2'
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=use_sqrt)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

                pred_final = pcds_pred[-1]
                cd_defect, cd_boundary = get_defect_boundary_cd(pred_final, gt, partial)
                defect_metrics.update([cd_defect, cd_boundary])

                # Compute F-score with threshold 0.01.
                if compute_f_score:
                    f_batch = Metrics.get_f_score_batch(pred_final, gt, th=0.01)
                    f_score_sum += f_batch * pred_final.shape[0]
                    f_score_count += pred_final.shape[0]

                if save_heatmap and heatmap_sample is None:
                    pred_final = pcds_pred[-1]
                    batch_size = pred_final.shape[0]
                    for i in range(batch_size):
                        if seen_samples == target_sample_index:
                            heatmap_sample = {
                                'partial': partial[i].detach().cpu().numpy(),
                                'gt': gt[i].detach().cpu().numpy(),
                                'pred': pred_final[i].detach().cpu().numpy(),
                                'model_id': str(model_id[i]),
                            }
                            break
                        seen_samples += 1
                
                pbar.set_postfix({'cd3': '{:.4f}'.format(cd3)})

        avg_f_score = f_score_sum / f_score_count if f_score_count > 0 else 0.0

        # Record testing results, including F-score.
        cd_defect_avg = defect_metrics.avg(0) if defect_metrics.count(0) > 0 else float('nan')
        cd_boundary_avg = defect_metrics.avg(1) if defect_metrics.count(1) > 0 else float('nan')

        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | #{:d} {:.4f}'.format(
            self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), 
            test_losses.avg(3), avg_f_score, cd_defect_avg, cd_boundary_avg, self.best_epoch, self.best_metrics)
        if record:
            self.test_record(message)
        if heatmap_sample is not None:
            self._save_training_heatmap(heatmap_sample)

        metrics_dict = {
            'cdc': test_losses.avg(0),
            'cd1': test_losses.avg(1),
            'cd2': test_losses.avg(2),
            'cd3': test_losses.avg(3),
            'partial_matching': test_losses.avg(4),
            'f_score': avg_f_score,
            'cd_defect': cd_defect_avg,
            'cd_boundary': cd_boundary_avg,
        }
        return test_losses.avg(3), metrics_dict, message


    def _print_training_summary(self):
        """Print mean and best values for each evaluation metric after training."""
        if not self.history:
            return
        epochs = [h['epoch'] for h in self.history]
        metrics_names = ['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching', 'f_score', 'cd_defect', 'cd_boundary']
        print('\n' + '=' * 70)
        print('Training complete - evaluation metric summary')
        print('=' * 70)
        for name in metrics_names:
            values = [h[name] for h in self.history]
            valid_vals = [v for v in values if not np.isnan(v)]
            mean_val = np.nanmean(values) if values else float('nan')
            if not valid_vals:
                best_val, best_epoch = float('nan'), '-'
            else:
                if name == 'f_score':
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)
                best_epoch = epochs[values.index(best_val)]
            print('  {:20s}: mean = {:.6f}, best = {:.6f} (Epoch {})'.format(name, mean_val, best_val, best_epoch))
        print('  Best Chamfer Distance (cd3) at Epoch {}: {:.6f}'.format(self.best_epoch, self.best_metrics))
        print('=' * 70 + '\n')

    def _plot_metrics_curves(self):
        """Plot each metric over epochs."""
        if not self.history:
            return
        out_dir = self.cfg.DIR.LOGS
        epochs = [h['epoch'] for h in self.history]
        metrics_config = [
            ('cdc', 'CDC (Chamfer)', False),
            ('cd1', 'CD1', False),
            ('cd2', 'CD2', False),
            ('cd3', 'CD3 (Chamfer Distance)', False),
            ('partial_matching', 'Partial Matching', False),
            ('f_score', 'F-score', True),
            ('cd_defect', 'CD-Defect', False),
            ('cd_boundary', 'CD-boundary', False),
        ]
        n_metrics = len(metrics_config)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        for idx, (key, label, greater_better) in enumerate(metrics_config):
            values = [h[key] for h in self.history]
            ax = axes[idx]
            ax.plot(epochs, values, 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title('{} vs Epoch'.format(label))
            ax.grid(True, alpha=0.3)
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        save_path = os.path.join(out_dir, 'metrics_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print('Metric curve figure saved to: {}'.format(save_path))

    def _save_metrics_excel(self):
        """Save epoch, Chamfer Distance, and F-score metrics to an Excel file."""
        if not self.history:
            return
        try:
            import pandas as pd
            rows = []
            for h in self.history:
                rows.append({
                    'Epoch': h['epoch'],
                    'Chamfer Distance': h['cd3'],
                    'F-score': h['f_score'],
                    'CD-Defect': h['cd_defect'],
                    'CD-boundary': h['cd_boundary'],
                })
            df = pd.DataFrame(rows)
            out_path = os.path.join(self.cfg.DIR.LOGS, 'training_metrics.xlsx')
            df.to_excel(out_path, index=False, engine='openpyxl')
            print('Excel metric file saved to: {}'.format(out_path))
        except ImportError as e:
            print('Tip: install pandas and openpyxl to save Excel files (pip install pandas openpyxl)')

    def _evaluate_per_category_fscore(self, model, val_data_loader, cfg):
        """Evaluate per-category mean F-score with the best checkpoint after training."""
        best_ckpt = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
        if not os.path.exists(best_ckpt):
            print('\n[Info] ckpt-best.pth was not found; skipping per-category F-score evaluation')
            return
        device = next(model.parameters()).device
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # PCN/ShapeNet category ID to name mapping.
        CATEGORY_NAMES = {
            '02691156': 'plane', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair',
            '03636649': 'lamp', '04256520': 'couch', '04379243': 'table', '04530566': 'watercraft'
        }
        category_f_scores = {}

        pbar = tqdm(val_data_loader, desc='Per-category F-score', ncols=100)
        with torch.no_grad():
            for taxonomy_ids, model_ids, data in pbar:
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']
                pcds_pred = model(partial.contiguous())
                pred_final = pcds_pred[-1]
                b = pred_final.shape[0]
                for i in range(b):
                    tid = taxonomy_ids[i]
                    tid_str = str(tid) if isinstance(tid, str) else str(tid.item())
                    f_i = Metrics._get_f_score(pred_final[i:i+1], gt[i:i+1], th=0.01)
                    category_f_scores.setdefault(tid_str, []).append(f_i)

        print('\n' + '=' * 50)
        print('Per-category mean F-score (best checkpoint)')
        print('=' * 50)
        for tid in sorted(category_f_scores.keys()):
            scores = category_f_scores[tid]
            mean_f = np.mean(scores)
            name = CATEGORY_NAMES.get(tid, tid)
            print('  {:12s} ({:s}): {:.4f}  (n={:d})'.format(name, tid, mean_f, len(scores)))
        all_scores = [f for scores in category_f_scores.values() for f in scores]
        overall_mean = np.mean(all_scores) if all_scores else 0.0
        print('  {:12s}: {:.4f}'.format('Overall', overall_mean))
        print('=' * 50 + '\n')

    def test(self, cfg, model=None, test_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        metric_names = ['CD-all', 'CD-Defect', 'CD-boundary']
        test_metrics = AverageMeter(metric_names)
        category_metrics = dict()
        category_models = dict()
        # F-score accumulation.
        f_score_sum = 0.0
        f_score_count = 0
        category_f_scores = dict()  # taxonomy_id -> [f_scores]

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                b, n, _ = partial.shape

                pcds_pred = model(partial.contiguous())
                use_sqrt = getattr(cfg.TEST, 'CD_TYPE', 'L2') == 'L2'
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=use_sqrt)


                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                pred_final = pcds_pred[-1]
                cd_defect, cd_boundary = get_defect_boundary_cd(pred_final, gt, partial)
                _metrics = [cd3, cd_defect, cd_boundary]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(metric_names)
                category_metrics[taxonomy_id].update(_metrics)

                # F-score based on the final predicted point cloud.
                f_batch = Metrics.get_f_score_batch(pred_final, gt, th=0.01)
                f_score_sum += f_batch * pred_final.shape[0]
                f_score_count += pred_final.shape[0]
                category_f_scores.setdefault(taxonomy_id, []).append(f_batch)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # record
                        if taxonomy_id not in category_models:
                            category_models[taxonomy_id] = []
                        category_models[taxonomy_id].append((model_name, cd3))
                        # Output comparison image: Complete / Partial / Prediction.
                        img_gt = pc_util.point_cloud_three_views(gt[mm, :].detach().cpu().numpy(), diameter=7)
                        img_pred = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        img_partial = pc_util.point_cloud_three_views(partial[mm, :].detach().cpu().numpy(), diameter=7)
                        # Row order: Complete (GT), Partial, Prediction.
                        comp = np.concatenate([img_gt, img_partial, img_pred], axis=0)
                        comp = (comp*255).astype('uint8')
                        im = Image.fromarray(comp)
                        # Add labels with a large Times-style font when available.
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(im)
                        h = img_gt.shape[0]
                        # Try Times New Roman first, then fall back to the default font.
                        try:
                            font = ImageFont.truetype("Times New Roman.ttf", size=40)
                        except Exception:
                            try:
                                font = ImageFont.truetype("Times.ttf", size=40)
                            except Exception:
                                font = ImageFont.load_default()
                        # Place large labels in the top-left corner of each panel.
                        margin_x = 60
                        margin_y = 50
                        draw.text((margin_x, margin_y), 'Complete', fill=(0, 0, 0), font=font)
                        draw.text((margin_x, h + margin_y), 'Partial', fill=(0, 0, 0), font=font)
                        draw.text((margin_x, 2 * h + margin_y), 'Prediction', fill=(0, 0, 0), font=font)

                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        im.save(img_filename)


        # Record category model cds
        if outdir:
            for cat in list(category_models.keys()):
                model_tuples = category_models[cat]
                with open(os.path.join(outdir, cat+'.txt'), 'w') as record_file:
                    # sort by cd
                    for tt in sorted(model_tuples, key=lambda x:x[1]):
                        record_file.write('{:s}  {:.4f}\n'.format(tt[0], tt[1])) 


        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            self.train_record(message)

        self.train_record('Overall\t\t' + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))

        # Record overall and per-category F-score.
        avg_f_score = f_score_sum / f_score_count if f_score_count > 0 else 0.0
        self.train_record('Overall F-score\t%.4f' % avg_f_score)
        for taxonomy_id in category_f_scores:
            scores = category_f_scores[taxonomy_id]
            mean_f = np.mean(scores) if scores else 0.0
            self.train_record('F-score\t{:s}\t{:.4f}'.format(str(taxonomy_id), mean_f))

        # record testing results
        overall_metrics = test_metrics.avg()
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | #{:d} {:.4f}'.format(
            self.epoch,
            test_losses.avg(0),
            test_losses.avg(1),
            test_losses.avg(2),
            test_losses.avg(4),
            test_losses.avg(3),
            avg_f_score,
            overall_metrics[1],
            overall_metrics[2],
            self.best_epoch,
            self.best_metrics,
        )
        self.test_record(message)


        return test_losses.avg(3)


class Manager_shapenet55:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # training dataset
        self.dataset = cfg.DATASET.TRAIN_DATASET

        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()

    def unpack_data(self, data):

        if self.dataset == 'ShapeNet':
            partial = data['partial_cloud']
            gt = data['gtcloud']
        elif self.dataset == 'ShapeNet55':
            # generate partial data online
            gt = data['gtcloud']
            _, npoints, _ = gt.shape
            partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        else:
            raise ValueError('No method implemented for this dataset: {:s}'.format(self.dataset))

        return partial, gt


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | f_score | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                pcds_pred = model(partial)

                loss_total, losses, gts = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

            # Validate the current model
            cd_eval = self.validate(cfg, model=model, val_data_loader=val_data_loader)
            self.train_record('Testing scores = {:.4f}'.format(cd_eval))

            # Update learning rate after optimizer.step() (PyTorch 1.1+ requirement)
            self.lr_scheduler.step()

            # Save checkpoints
            if cd_eval < self.best_metrics:
                self.best_epoch = epoch_idx
                file_name = 'ckpt-best.pth' if cd_eval < self.best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None, val_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter('cd3')

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                use_sqrt = getattr(cfg.TEST, 'CD_TYPE', 'L2') == 'L2'
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=use_sqrt)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message, show_info=False)

        return test_losses.avg(3)

    def test(self, cfg, model, test_data_loader, outdir, mode=None):

        if self.dataset == 'ShapeNet':
            self.test_pcn(cfg, model, test_data_loader, outdir)
        elif self.dataset == 'ShapeNet55':
            self.test_shapenet55(cfg, model, test_data_loader, outdir, mode)
        else:
            raise ValueError('No testing method implemented for this dataset: {:s}'.format(self.dataset))

    def test_pcn(self, cfg, model=None, test_data_loader=None, outdir=None):
        """
        Testing Method for dataset PCN
        """

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                # get loss
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                # get all metrics
                _metrics = Metrics.get(pcds_pred[-1], gt)
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)

    def test_shapenet55(self, cfg, model=None, test_data_loader=None, outdir=None, mode=None):
        """
        Testing Method for dataset shapenet-55/34
        """

        from models.modules.utils import fps_subsample
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Eval settings
        crop_ratio = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }
        choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                gt = data['gtcloud']
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)

                    pcds_pred = model(partial.contiguous())
                    loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=False) # L2

                    # get loss
                    cdc = losses[0].item() * 1e3
                    cd1 = losses[1].item() * 1e3
                    cd2 = losses[2].item() * 1e3
                    cd3 = losses[3].item() * 1e3
                    partial_matching = losses[4].item() * 1e3
                    test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                    # get all metrics
                    _metrics = Metrics.get(pcds_pred[-1], gt)
                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)

                    # output to file
                    if outdir:
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                        # save pred, gt, partial pcds 
                        pred = pcds_pred[-1]
                        for mm, model_name in enumerate(model_id):
                            output_file = os.path.join(outdir, taxonomy_id, model_name+'_{:02d}'.format(partial_id))
                            write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            # output img files
                            img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                            output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                            output_img = (output_img*255).astype('uint8')
                            im = Image.fromarray(output_img)
                            im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)


class Manager_MVP:
    def __init__(self, model, cfg):


        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # self.lr_decays = {i: 0.1 ** (1 / cfg.TRAIN.LR_DECAY) for i in range(1, cfg.TRAIN.N_EPOCHS+1)}
        # self.scheduler_steplr = StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=self.scheduler_steplr)
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | f_score | #best_epoch best_metrics')
        cd_best  = 1e6
        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (label, partial, gt) in enumerate(train_data_loader):
                #for k, v in data.items():
                #    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = partial.float().cuda()
                gt = gt.float().cuda()

                pcds_pred = model(partial)

                loss_total, losses, gts = get_loss_mvp(pcds_pred, partial, gt, sqrt=True)

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

                # if steps <= cfg.TRAIN.WARMUP_STEPS:
                #     self.lr_scheduler.step()
                #     steps += 1

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # Training record

            # save checkpoint 
            now_cd = self.validate(cfg, model, val_data_loader)

            # Update learning rate after optimizer.step() (PyTorch 1.1+ requirement)
            self.lr_scheduler.step()

            if now_cd < cd_best:
                cd_best = now_cd
                file_name = 'ckpt-best.pth' 
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'model': model.state_dict()
                }, output_path)
            
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s cd_best = %f' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]], cd_best))
        # training end
        self.train_record_file.close()
        self.test_record_file.close()

    def validate(self, cfg, model=None, val_data_loader=None):


        metrics = ['cd_p', 'cd_t', 'f1']
        test_loss_meters = {m: AverageValueMeter() for m in metrics}
        model.eval()

        with torch.no_grad():
            for data in val_data_loader:
                label, inputs_cpu, gt_cpu = data

                inputs = inputs_cpu.float().cuda()
                gt = gt_cpu.float().cuda()
                
                output = model(inputs)[-1]
                cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)
                result_dict = dict()
                result_dict['cd_p'] = cd_p
                result_dict['cd_t'] = cd_t
                result_dict['f1'] = f1

                for k, v in test_loss_meters.items():
                    v.update(result_dict[k].mean().item())
                
            
        return test_loss_meters['cd_t'].avg
