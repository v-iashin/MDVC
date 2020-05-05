import os
import argparse
from time import strftime, localtime
from shutil import copytree, ignore_patterns

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard as tensorboard
# import tensorboardX as tensorboard

from model.transformer import SubsAudioVideoTransformer
from dataset.dataset import ActivityNetCaptionsIteratorDataset
from loss.loss import LabelSmoothing, SimpleLossCompute
from scheduler.lr_scheduler import SimpleScheduler
from epoch_loop.run_epoch import training_loop, validation_next_word_loop, greedy_decoder
from epoch_loop.run_epoch import save_model, validation_1by1_loop, average_metrics_in_two_dicts
from utils.utils import timer

class Config(object):
    '''
    Note: don't change the methods of this class later in code.
    '''
    
    def __init__(self, args):
        '''
        Try not to create anything here: like new forders or something
        '''
        self.curr_time = strftime('%y%m%d%H%M%S', localtime())
        # dataset
        self.train_meta_path = args.train_meta_path
        self.val_1_meta_path = args.val_1_meta_path
        self.val_2_meta_path = args.val_2_meta_path
        self.val_prop_meta_path = args.val_prop_meta_path
        self.modality = args.modality
        self.video_feature_name = args.video_feature_name
        self.video_features_path = args.video_features_path
        self.filter_video_feats = args.filter_video_feats
        self.average_video_feats = args.average_video_feats
        self.audio_feature_name = args.audio_feature_name
        self.audio_features_path = args.audio_features_path
        self.filter_audio_feats = args.filter_audio_feats
        self.average_audio_feats = args.average_audio_feats
        self.use_categories = args.use_categories
        if self.use_categories:
            self.video_categories_meta_path = args.video_categories_meta_path
        # make them d_video and d_audio
        self.d_vid = args.d_vid
        self.d_aud = args.d_aud
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.max_len = args.max_len
        self.min_freq = args.min_freq
        # model
        self.model = args.model
        self.dout_p = args.dout_p
        self.N = args.N
        self.use_linear_embedder = args.use_linear_embedder
        if args.use_linear_embedder:
            self.d_model_video = args.d_model_video
            self.d_model_audio = args.d_model_audio
        else:
            self.d_model_video = self.d_vid
            self.d_model_audio = self.d_aud
        self.d_model_subs = args.d_model_subs
        if self.model == 'transformer':
            self.H = args.H
            self.d_ff_video = args.d_ff_video
            self.d_ff_audio = args.d_ff_audio
            self.d_ff_subs = args.d_ff_subs
            if self.use_categories:
                self.d_cat = args.d_cat
        elif self.model == 'bi_gru':
            pass
        else:
            raise Exception(f'Undefined model: "{self.model}"')
            
        # training
        self.device_ids = args.device_ids
        self.device = f'cuda:{self.device_ids[0]}'
        self.train_batch_size = args.B * len(self.device_ids)
        self.inference_batch_size = args.inf_B_coeff * self.train_batch_size
        self.start_epoch = args.start_epoch # todo: pretraining
        self.epoch_num = args.epoch_num
        self.one_by_one_starts_at = args.one_by_one_starts_at
        self.early_stop_after = args.early_stop_after
        # criterion
        self.criterion = args.criterion
        self.smoothing = args.smoothing # 0 == cross entropy
        # optimizer
        self.optimizer = args.optimizer
        if self.optimizer == 'adam':
            self.beta1, self.beta2 = args.betas
            self.eps = args.eps
        else:
            raise Exception(f'Undefined optimizer: "{self.optimizer}"')
        # lr scheduler
        self.scheduler = args.scheduler
        if self.scheduler == 'attention_is_all_you_need':
            self.lr_coeff = args.lr_coeff
            self.warmup_steps = args.warmup_steps
        elif self.scheduler == 'constant':
            self.lr = args.lr
        else:
            raise Exception(f'Undefined scheduler: "{self.scheduler}"')
        # evaluation
        self.reference_paths = args.reference_paths
        self.tIoUs = args.tIoUs
        self.max_prop_per_vid = args.max_prop_per_vid
        self.verbose_evaluation = args.verbose_evaluation
        # logging
        self.to_log = args.to_log
        self.videos_to_monitor = args.videos_to_monitor
        if args.to_log:
            self.log_dir = args.log_dir
            self.checkpoint_dir = self.log_dir # the same yes
            exper_name = self.make_experiment_name()
            self.comment = args.comment
            self.log_path = os.path.join(self.log_dir, exper_name)
            self.model_checkpoint_path = os.path.join(self.checkpoint_dir, exper_name)
        else:
            self.log_dir = None
            self.log_path = None
        
    def make_experiment_name(self):
        return self.curr_time[2:]

    def get_params(self, out_type):
        
        if out_type == 'md_table':
            table  = '| Parameter | Value | \n'
            table += '|-----------|-------| \n'

            for par, val in vars(self).items():
                table += f'| {par} | {val}| \n'

            return table
        
        elif out_type == 'dict':
            params_to_filter = [
                'model_checkpoint_path', 'log_path', 'comment', 'curr_time', 
                'checkpoint_dir', 'log_dir', 'videos_to_monitor', 'to_log', 
                'verbose_evaluation', 'tIoUs', 'reference_paths', 
                'one_by_one_starts_at', 'device', 'device_ids', 'pad_token',
                'end_token', 'start_token', 'val_1_meta_path', 'video_feature_name',
                'val_2_meta_path', 'train_meta_path', 'betas', 'path'
            ]
            dct = vars(self)
            dct = {k: v for k, v in dct.items() if (k not in params_to_filter) and (v is not None)}
            
            return dct
    
    def self_copy(self):
        
        if self.to_log:
            # let it be in method's arguments (for TBoard)
            self.path = os.path.realpath(__file__)
            pwd = os.path.split(self.path)[0]
            cp_path = os.path.join(self.model_checkpoint_path, 'wdir_copy')
            copytree(pwd, cp_path, ignore=ignore_patterns('todel', 'submodules', '.git'))


def main(cfg):
    ###########################################################################
    ######################### Some reminders to print #########################
    ###########################################################################
    if cfg.to_log:
        print(f'log_path: {cfg.log_path}')
        print(f'model_checkpoint_path: {cfg.model_checkpoint_path}')
    ###########################################################################
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.cuda.set_device(cfg.device_ids[0])

    train_dataset = ActivityNetCaptionsIteratorDataset(
        cfg.start_token, cfg.end_token, cfg.pad_token, cfg.min_freq, 
        cfg.train_batch_size, cfg.video_features_path, cfg.video_feature_name, 
        cfg.filter_video_feats, cfg.average_video_feats,
        cfg.audio_features_path, cfg.audio_feature_name, 
        cfg.filter_audio_feats, cfg.average_audio_feats, 
        cfg.train_meta_path, cfg.val_1_meta_path, 
        cfg.val_2_meta_path, torch.device(cfg.device), 'train', cfg.modality, 
        cfg.use_categories, props_are_gt=True, get_full_feat=False
    )
    val_1_dataset = ActivityNetCaptionsIteratorDataset(
        cfg.start_token, cfg.end_token, cfg.pad_token, cfg.min_freq, 
        cfg.inference_batch_size, cfg.video_features_path, cfg.video_feature_name, 
        cfg.filter_video_feats, cfg.average_video_feats, 
        cfg.audio_features_path, cfg.audio_feature_name, 
        cfg.filter_audio_feats, cfg.average_audio_feats,  cfg.train_meta_path, cfg.val_1_meta_path, 
        cfg.val_2_meta_path, torch.device(cfg.device), 'val_1', cfg.modality, 
        cfg.use_categories, props_are_gt=True, get_full_feat=False
    )
    val_2_dataset = ActivityNetCaptionsIteratorDataset(
        cfg.start_token, cfg.end_token, cfg.pad_token, cfg.min_freq, 
        cfg.inference_batch_size, cfg.video_features_path, cfg.video_feature_name, 
        cfg.filter_video_feats, cfg.average_video_feats, 
        cfg.audio_features_path, cfg.audio_feature_name, 
        cfg.filter_audio_feats, cfg.average_audio_feats, cfg.train_meta_path, cfg.val_1_meta_path, 
        cfg.val_2_meta_path, torch.device(cfg.device), 'val_2', cfg.modality, 
        cfg.use_categories, props_are_gt=True, get_full_feat=False
    )
    # 'val_1' in phase doesn't really matter because props are for validation set
    # cfg.val_1_meta_path -> cfg.val_prop_meta
    val_pred_prop_dataset = ActivityNetCaptionsIteratorDataset(
        cfg.start_token, cfg.end_token, cfg.pad_token, cfg.min_freq, 
        cfg.inference_batch_size, cfg.video_features_path, cfg.video_feature_name, 
        cfg.filter_video_feats, cfg.average_video_feats, 
        cfg.audio_features_path, cfg.audio_feature_name, 
        cfg.filter_audio_feats, cfg.average_audio_feats, cfg.train_meta_path, 
        cfg.val_prop_meta_path, 
        cfg.val_2_meta_path, torch.device(cfg.device), 'val_1', cfg.modality, 
        cfg.use_categories, props_are_gt=False, get_full_feat=False
    )
    
    # make sure that DataLoader has batch_size = 1!
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.dont_collate)
    val_1_loader = DataLoader(val_1_dataset, collate_fn=val_1_dataset.dont_collate)
    val_2_loader = DataLoader(val_2_dataset, collate_fn=val_2_dataset.dont_collate)
    val_pred_prop_loader = DataLoader(val_pred_prop_dataset, collate_fn=val_2_dataset.dont_collate)
    
    model = SubsAudioVideoTransformer(
        train_dataset.trg_voc_size, train_dataset.subs_voc_size,
        cfg.d_aud, cfg.d_vid, cfg.d_model_audio, cfg.d_model_video,
        cfg.d_model_subs,
        cfg.d_ff_audio, cfg.d_ff_video, cfg.d_ff_subs,
        cfg.N, cfg.N, cfg.N, cfg.dout_p, cfg.H, cfg.use_linear_embedder
    )
    
    criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    
    # lr = 0 here have no impact on training (see lr scheduler)
    optimizer = torch.optim.Adam(
        model.parameters(), 0, (cfg.beta1, cfg.beta2), cfg.eps
    )
    lr_scheduler = SimpleScheduler(optimizer, cfg.lr)
    loss_compute = SimpleLossCompute(criterion, lr_scheduler)

    model.to(torch.device(cfg.device))
    # haven't tested for multi GPU for a while -- might not work. 
    model = torch.nn.DataParallel(model, cfg.device_ids)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Param Num: {param_num}')
    
    if cfg.to_log:
        os.makedirs(cfg.log_path)
        os.makedirs(cfg.model_checkpoint_path, exist_ok=True) # handles the case when model_checkpoint_path = log_path
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_text('config', cfg.get_params('md_table'), 0)
        TBoard.add_text('config/comment', cfg.comment, 0)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0

    for epoch in range(cfg.start_epoch, cfg.epoch_num):
        num_epoch_best_metric_unchanged += 1
        
        if (num_epoch_best_metric_unchanged == cfg.early_stop_after) or (timer(cfg.curr_time) > 67):
            print(f'Early stop at {epoch}: unchanged for {num_epoch_best_metric_unchanged} epochs')
            print(f'Current timer: {timer(cfg.curr_time)}')
            break
        
        # train
        training_loop(
            model, train_loader, loss_compute, lr_scheduler, epoch, TBoard, 
            cfg.modality, cfg.use_categories
        )
        # validation (next word)
        val_1_loss = validation_next_word_loop(
            model, val_1_loader, greedy_decoder, loss_compute, lr_scheduler, 
            epoch, cfg.max_len, cfg.videos_to_monitor, TBoard, cfg.modality, 
            cfg.use_categories
        )
        val_2_loss = validation_next_word_loop(
            model, val_2_loader, greedy_decoder, loss_compute, lr_scheduler, 
            epoch, cfg.max_len, cfg.videos_to_monitor, TBoard, cfg.modality, 
            cfg.use_categories
        )
        
        val_loss_avg = (val_1_loss + val_2_loss) / 2
        
        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at:
            # validation with g.t. proposals
            val_1_metrics = validation_1by1_loop(
                model, val_1_loader, greedy_decoder, loss_compute, lr_scheduler, 
                epoch, cfg.max_len, cfg.log_path, 
                cfg.verbose_evaluation, [cfg.reference_paths[0]], cfg.tIoUs, 
                cfg.max_prop_per_vid, TBoard, cfg.modality, cfg.use_categories, 
            )
            val_2_metrics = validation_1by1_loop(
                model, val_2_loader, greedy_decoder, loss_compute, lr_scheduler, 
                epoch, cfg.max_len, cfg.log_path, 
                cfg.verbose_evaluation, [cfg.reference_paths[1]], cfg.tIoUs, 
                cfg.max_prop_per_vid, TBoard, cfg.modality, cfg.use_categories, 
            )
            
            if cfg.to_log:
                # averaging metrics obtained from val_1 and val_2
                metrics_avg = average_metrics_in_two_dicts(val_1_metrics, val_2_metrics)
                metrics_avg = metrics_avg['Average across tIoUs']
                
                TBoard.add_scalar('metrics/val_loss_avg', val_loss_avg, epoch)
                TBoard.add_scalar('metrics/meteor', metrics_avg['METEOR'] * 100, epoch)
                TBoard.add_scalar('metrics/bleu4', metrics_avg['Bleu_4'] * 100, epoch)
                TBoard.add_scalar('val_avg/bleu3', metrics_avg['Bleu_3'] * 100, epoch)
                TBoard.add_scalar('val_avg/bleu2', metrics_avg['Bleu_2'] * 100, epoch)
                TBoard.add_scalar('val_avg/bleu1', metrics_avg['Bleu_1'] * 100, epoch)
                TBoard.add_scalar('val_avg/rouge_l', metrics_avg['ROUGE_L'] * 100, epoch)
                TBoard.add_scalar('val_avg/cider', metrics_avg['CIDEr'] * 100, epoch)
                TBoard.add_scalar('val_avg/precision', metrics_avg['Precision'] * 100, epoch)
                TBoard.add_scalar('val_avg/recall', metrics_avg['Recall'] * 100, epoch)
            
                # saving the model if it is better than the best so far
                if best_metric < metrics_avg['METEOR']:
                    best_metric = metrics_avg['METEOR']
                    
                    save_model(
                        cfg, epoch, model, optimizer, val_1_loss, val_2_loss, 
                        val_1_metrics, val_2_metrics, train_dataset.trg_voc_size
                    )
                    # reset the early stopping criterion
                    num_epoch_best_metric_unchanged = 0
                    
                # put it after: so on zeroth epoch it is not zero
                TBoard.add_scalar('val_avg/best_metric_meteor', best_metric * 100, epoch)
                
    if cfg.to_log:
        # load the best model
        best_model_path = os.path.join(cfg.model_checkpoint_path, 'best_model.pt')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_metrics_pred_prop = validation_1by1_loop(
            model, val_pred_prop_loader, greedy_decoder, loss_compute, lr_scheduler, 
            checkpoint['epoch'], cfg.max_len, cfg.log_path, 
            cfg.verbose_evaluation, cfg.reference_paths, cfg.tIoUs, 
            cfg.max_prop_per_vid, TBoard, cfg.modality, cfg.use_categories
        )
        best_metric_pred_prop = val_metrics_pred_prop['Average across tIoUs']['METEOR']
        print(f'best_metric: {best_metric}')
        print(f'best_metric_pred_prop: {best_metric_pred_prop}')
        TBoard.close()
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run experiment')
    
    parser.add_argument(
        '--train_meta_path', type=str, default='./data/train_meta.csv', 
        help='path to the precalculated train meta file'
    )
    parser.add_argument(
        '--val_1_meta_path', type=str, default='./data/val_1_meta.csv', 
        help='path to the precalculated val 1 meta file'
    )
    parser.add_argument(
        '--val_2_meta_path', type=str, default='./data/val_2_meta.csv', 
        help='path to the precalculated val 2 meta file'
    )
    parser.add_argument(
        '--val_prop_meta_path', type=str, default='./data/bafcg_val_100_proposal_result.csv', 
        help='path to the precalculated proposals on the validation set'
    )
    parser.add_argument(
        '--dont_log', dest='to_log', action='store_false', 
        help='Prevent logging in the experiment.'
    )
    parser.add_argument(
        '--device_ids', type=int, nargs='+', required=True,
        help='device indices separated by a whitespace'
    )
    parser.add_argument(
        '--use_categories', dest='use_categories', action='store_true', 
        help='whether to condition the model on categories'
    )
    parser.add_argument(
        '--video_categories_meta_path', type=str, default='./data/videoCategoriesMetaUS.json',
        help='Path to the categories meta from Youtube API: \
        https://developers.google.com/youtube/v3/docs/videoCategories/list'
    )
    parser.add_argument(
        '--d_cat', type=int,
        help='size of the category embedding layer'
    )
    parser.add_argument(
        '--modality', type=str, default='subs_audio_video',
        choices=['audio', 'video', 'audio_video', 'subs_audio_video'],
    )
    parser.add_argument('--video_feature_name', type=str, default='i3d')
    parser.add_argument(
        '--video_features_path', type=str, 
        default='./data/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5'
    )
    parser.add_argument('--audio_feature_name', type=str, default='vggish')
    parser.add_argument(
        '--audio_features_path', type=str, default='./data/sub_activitynet_v1-3.vggish.hdf5'
    )
    parser.add_argument('--d_vid', type=int, default=1024)
    parser.add_argument('--d_aud', type=int, default=128)
    parser.add_argument(
        '--filter_video_feats', dest='filter_video_feats', action='store_true', 
        help='filter video features (removes overlap 16/8 -> 16/16).'
    )
    parser.add_argument(
        '--average_video_feats', dest='average_video_feats', action='store_true', 
        help='averages video features (designed for c3d: 16x4 -> 16 (the same time span)).'
    )
    parser.add_argument(
        '--filter_audio_feats', dest='filter_audio_feats', action='store_true', 
        help='filter video features (removes overlap 16/8 -> 16/16).'
    )
    parser.add_argument(
        '--average_audio_feats', dest='average_audio_feats', action='store_true', 
        help='averages audio features.'
    )
    parser.add_argument(
        '--start_token', type=str, default='<s>',
        help='starting token'
    )
    parser.add_argument(
        '--end_token', type=str, default='</s>',
        help='ending token'
    )
    parser.add_argument(
        '--pad_token', type=str, default='<blank>',
        help='padding token'
    )
    parser.add_argument(
        '--max_len', type=int, default=50,
        help='maximum size of 1by1 prediction'
    )
    parser.add_argument(
        '--min_freq', type=int, default=1,
        help='to be in the vocab a word should appear min_freq times in train dataset'
    )
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--dout_p', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=1, help='number of layers in a model')
    parser.add_argument(
        '--use_linear_embedder', dest='use_linear_embedder', action='store_true', 
        help='Whether to include a dense layer between vid features and RNN'
    )
    parser.add_argument(
        '--d_model_video', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--d_model_audio', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for audio model'
    )
    parser.add_argument('--d_model_subs', type=int, default=512)
    parser.add_argument(
        '--H', type=int, default=4,
        help='number of heads in multiheaded attention in Transformer'
    )
    parser.add_argument(
        '--d_ff_video', type=int, default=2048,
        help='size of the internal layer of PositionwiseFeedForward net in Transformer (Video)'
    )
    parser.add_argument(
        '--d_ff_audio', type=int, default=2048,
        help='size of the internal layer of PositionwiseFeedForward net in Transformer (Audio)'
    )
    parser.add_argument(
        '--d_ff_subs', type=int, default=2048,
        help='size of the internal layer of PositionwiseFeedForward net in Transformer (Subs)'
    )
    parser.add_argument(
        '--B', type=int, default=28,
        help='batch size per a device'
    )
    parser.add_argument(
        '--inf_B_coeff', type=int, default=2,
        help='the batch size on inference is inf_B_coeff times the B'
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0, choices=[0],
        help='the epoch number to start training (if specified, pretraining a net from start_epoch epoch)'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=45,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--one_by_one_starts_at', type=int, default=0,
        help='number of epochs to skip before starting 1-by-1 validation'
    )
    parser.add_argument(
        '--early_stop_after', type=int, default=50,
        help='number of epochs to wait for best metric to change before stopping'
    )
    parser.add_argument(
        '--criterion', type=str, default='label_smoothing', choices=['label_smoothing'],
        help='criterion to measure the loss'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.7,
        help='smoothing coeff (= 0 cross ent loss; -> 1 more smoothing, random labels) must be in [0, 1]'
    )
    parser.add_argument(
        '--optimizer', type=str, default='adam', choices=['adam'],
        help='optimizer'
    )
    parser.add_argument(
        '--betas', type=float, nargs=2, default=[0.9, 0.98],
        help='beta 1 and beta 2 parameters in adam'
    )
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='eps parameter in adam'
    )
    parser.add_argument(
        '--scheduler', type=str, default='constant', choices=['attention_is_all_you_need', 'constant'], 
        help='lr scheduler'
    )
    parser.add_argument(
        '--lr_coeff', type=float,  
        help='lr scheduler coefficient (if scheduler is attention_is_all_you_need)'
    )
    parser.add_argument(
        '--warmup_steps', type=int,
        help='number of "warmup steps" (if scheduler is attention_is_all_you_need)'
    )
    parser.add_argument('--lr', type=float, default=1e-5, help='lr (if scheduler is constant)')
    parser.add_argument(
        '--reference_paths', type=str, default=['./data/val_1.json', './data/val_2.json'], 
        nargs='+',
        help='reference paths for 1-by-1 validation'
    )
    parser.add_argument(
        '--tIoUs', type=float, default=[0.3, 0.5, 0.7, 0.9], nargs='+',
        help='thresholds for tIoU to be used for 1-by-1 validation'
    )
    parser.add_argument(
        '--max_prop_per_vid', type=int, default=1000,
        help='max number of proposal to take into considetation for 1-by-1 validation'
    )
    parser.add_argument(
        '--dont_verbose_evaluation', dest='verbose_evaluation', action='store_false', 
        help='dont verbose the evaluation server in 1-by-1 validation (no Precision and R)'
    )
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument(
        '--videos_to_monitor', type=str, nargs='+', 
        default=['v_GGSY1Qvo990', 'v_bXdq2zI1Ms0', 'v_aLv03Fznf5A'],
        help='the videos to monitor on validation loop with 1 by 1 prediction'
    )
    parser.add_argument('--comment', type=str, default='', help='comment for the experiment')

    parser.set_defaults(to_log=True)
    parser.set_defaults(filter_video_feats=False)
    parser.set_defaults(average_video_feats=False)
    parser.set_defaults(filter_audio_feats=False)
    parser.set_defaults(average_audio_feats=False)
    parser.set_defaults(use_linear_embedder=False)
    parser.set_defaults(verbose_evaluation=True)
    parser.set_defaults(use_categories=False)
    
    args = parser.parse_args()
    # print(args)
    cfg = Config(args)
    main(cfg)
