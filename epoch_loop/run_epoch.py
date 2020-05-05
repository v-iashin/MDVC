import os
import json
from tqdm import tqdm
import numpy as np
import torch
import spacy
from time import time, strftime, localtime

from model.transformer import mask
from evaluate.evaluate import ANETcaptions
from dataset.dataset import load_multimodal_features_from_h5, filter_features
from utils.utils import HiddenPrints

def calculate_metrics(reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose):
    metrics = {}
    PREDICTION_FIELDS = ['results', 'version', 'external_data']
    evaluator = ANETcaptions(
        reference_paths, submission_path, tIoUs, 
        max_prop_per_vid, PREDICTION_FIELDS, verbose)
    evaluator.evaluate()
    
    for i, tiou in enumerate(tIoUs):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    # Print the averages
    
    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))
    
    return metrics

def average_metrics_in_two_dicts(val_1_metrics, val_2_metrics):
    '''
        both dicts must have the same keys
    '''
    val_metrics_avg = {}
    
    for key in val_1_metrics.keys():
        val_metrics_avg[key] = {}
        
        for metric_name in val_1_metrics[key].keys():
            val_1_metric = val_1_metrics[key][metric_name]
            val_2_metric = val_2_metrics[key][metric_name]
            val_metrics_avg[key][metric_name] = (val_1_metric + val_2_metric) / 2
            
    return val_metrics_avg

def greedy_decoder(model, src, max_len, start_idx, end_idx, pad_idx, modality, 
                   categories=None):
    
    if modality == 'audio_video':
        assert model.training == False, 'call model.eval first'
    
        # src_video (video features): (B, S, d_feat_vid) ex: [3, 11, 1024], 
        # src_audio (video features): (B, S, d_feat_aud) ex: [3, 11, 128], 
        src_video, src_audio = src

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(len(src_video), 1).byte().to(src_video.device)

        with torch.no_grad():
            B, S = src_audio.size(0), src_audio.size(1)
            trg = (torch.ones(B, 1) * start_idx).type_as(src_audio).long()

            while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
                masks = mask(src_video[:, :, 0], trg, pad_idx)
                if categories is not None:
                    preds = model(src, trg, masks, categories)
                else:
                    preds = model(src, trg, masks)
                next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
                trg = torch.cat([trg, next_word], dim=-1)

                # sum two masks (or adding 1s where the ending token occured)
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg
    
    elif modality == 'subs_audio_video':
        assert model.training == False, 'call model.eval first'
    
        # src_video (video features): (B, S, d_feat_vid) ex: [3, 11, 1024], 
        # src_audio (video features): (B, S, d_feat_aud) ex: [3, 11, 128],
        # src_audio (video features): (B, Ss, d_model_subs) ex: [3, 14, 512],
        src_video, src_audio, src_subs = src

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(len(src_video), 1).byte().to(src_video.device)

        with torch.no_grad():
            B, S = src_audio.size(0), src_audio.size(1)
            trg = (torch.ones(B, 1) * start_idx).type_as(src_audio).long()

            while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
                src_mask, trg_mask = mask(src_video[:, :, 0], trg, pad_idx)
                src_subs_mask = mask(src_subs, None, pad_idx)
                masks = src_mask, trg_mask, src_subs_mask
                if categories is not None:
                    preds = model(src, trg, masks, categories)
                else:
                    preds = model(src, trg, masks)
                next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
                trg = torch.cat([trg, next_word], dim=-1)

                # sum two masks (or adding 1s where the ending token occured)
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg
    
    else:
        assert model.training == False, 'call model.eval first'

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(len(src), 1).byte().to(src.device)

        with torch.no_grad():
            # src (text): (B, S); src (video features): (B, S, D) [3, 11, 1024]
            B, S = src.size(0), src.size(1)
            trg = (torch.ones(B, 1) * start_idx).type_as(src).long()

            while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
                masks = mask(src[:, :, 0], trg, pad_idx)
                preds = model(src, trg, masks)
                next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
                trg = torch.cat([trg, next_word], dim=-1)

                # sum two masks (or adding 1s where the ending token occured)
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg
    
def encode_subs(train_subs_vocab, idx, meta, start_idx, end_idx):
    subs = meta.iloc[idx]['subs']
    # check for 'nan'
    if subs != subs:
        subs = ''
    subs = [token.text for token in spacy.load('en').tokenizer(subs)]
    subs = [train_subs_vocab.stoi[word] for word in subs]
    subs = [start_idx] + subs + [end_idx]
    return torch.tensor(subs)

def predict_1by1_for_TBoard(vid_ids_list, val_loader, decoder, model, max_len, 
                            modality, use_categories=None):
    meta = val_loader.dataset.features_dataset.dataset
    feature_names = val_loader.dataset.feature_names
    device = val_loader.dataset.device
    start_idx = val_loader.dataset.start_idx
    end_idx = val_loader.dataset.end_idx
    pad_idx = val_loader.dataset.pad_idx
    
    text = ''

    for vid_id in vid_ids_list:
        meta_subset = meta[meta['video_id'] == vid_id]
        text += f'\t {vid_id} \n'

        for (video_id, cap, start, end, duration, category, subs, phase, idx) in meta_subset.values:
            
            if modality == 'audio_video':
                feat_h5_audio = val_loader.dataset.features_dataset.feat_h5_audio
                feat_h5_video = val_loader.dataset.features_dataset.feat_h5_video
                feature_names_list = val_loader.dataset.features_dataset.feature_names_list
                filter_audio_feats = val_loader.dataset.features_dataset.filter_audio_feats
                filter_video_feats = val_loader.dataset.features_dataset.filter_video_feats
                average_audio_feats = val_loader.dataset.features_dataset.average_audio_feats
                average_video_feats = val_loader.dataset.features_dataset.average_video_feats
                # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
                video_stack_rgb, video_stack_flow, audio_stack = load_multimodal_features_from_h5(
                    feat_h5_video, feat_h5_audio, feature_names_list, video_id, start, end, duration
                )
                if filter_video_feats:
                    video_stack_rgb = filter_features(video_stack_rgb, average_video_feats)
                    video_stack_flow = filter_features(video_stack_flow, average_video_feats)
                if filter_audio_feats:
                    audio_stack = filter_features(audio_stack, average_audio_feats)
                    
                video_stack_rgb = video_stack_rgb.unsqueeze(0).to(device)
                video_stack_flow = video_stack_flow.unsqueeze(0).to(device)
                audio_stack = audio_stack.unsqueeze(0).to(device)
                stack = video_stack_rgb + video_stack_flow, audio_stack
            
            elif modality == 'subs_audio_video':
                feat_h5_audio = val_loader.dataset.features_dataset.feat_h5_audio
                feat_h5_video = val_loader.dataset.features_dataset.feat_h5_video
                feature_names_list = val_loader.dataset.features_dataset.feature_names_list
                filter_audio_feats = val_loader.dataset.features_dataset.filter_audio_feats
                filter_video_feats = val_loader.dataset.features_dataset.filter_video_feats
                average_audio_feats = val_loader.dataset.features_dataset.average_audio_feats
                average_video_feats = val_loader.dataset.features_dataset.average_video_feats
                train_subs_vocab = val_loader.dataset.train_subs_vocab
                # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
                video_stack_rgb, video_stack_flow, audio_stack = load_multimodal_features_from_h5(
                    feat_h5_video, feat_h5_audio, feature_names_list, video_id, start, end, duration
                )
                if filter_video_feats:
                    video_stack_rgb = filter_features(video_stack_rgb, average_video_feats)
                    video_stack_flow = filter_features(video_stack_flow, average_video_feats)
                if filter_audio_feats:
                    audio_stack = filter_features(audio_stack, average_audio_feats)

                subs_stack = encode_subs(train_subs_vocab, idx, meta, start_idx, end_idx)
                
                video_stack_rgb = video_stack_rgb.unsqueeze(0).to(device)
                video_stack_flow = video_stack_flow.unsqueeze(0).to(device)
                audio_stack = audio_stack.unsqueeze(0).to(device)
                subs_stack = subs_stack.unsqueeze(0).to(device)
                stack = video_stack_rgb + video_stack_flow, audio_stack, subs_stack
            
            else:
                raise NotImplementedError
            
            if use_categories:
                category = torch.tensor([category]).unsqueeze(0).to(device)
                trg_ints = decoder(
                    model, stack, max_len, start_idx, end_idx, pad_idx, modality, 
                    category
                )
            else:
                trg_ints = decoder(
                    model, stack, max_len, start_idx, end_idx, pad_idx, modality
                )
            trg_ints = trg_ints.cpu().numpy()[0]
            trg_words = [val_loader.dataset.train_vocab.itos[i] for i in trg_ints]
            en_sent = ' '.join(trg_words)

            text += f'\t P sent: {en_sent} \n'
            text += f'\t P proposals: {start//60:.0f}:{start%60:02.0f} {end//60:.0f}:{end%60:02.0f} '
            text += f'link: https://www.youtube.com/embed/{vid_id[2:]}?start={start}&end={end}&rel=0 \n'
            
        text += '\t \n'
    
    return text

def save_model(cfg, epoch, model, optimizer, val_1_loss_value, val_2_loss_value, 
               val_1_metrics, val_2_metrics, trg_voc_size):
    
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_1_loss': val_1_loss_value,
        'val_2_loss': val_2_loss_value,
        'val_1_metrics': val_1_metrics,
        'val_2_metrics': val_2_metrics,
        'trg_voc_size': trg_voc_size,
    }
    
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
    
#     path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_model.pt')
    torch.save(dict_to_save, path_to_save)

def training_loop(model, loader, loss_compute, lr_scheduler, epoch, TBoard, 
                  modality, use_categories):
    model.train()
    losses = []
    
    loader.dataset.update_iterator()
    feature_names = loader.dataset.feature_names
    
    time = strftime('%X', localtime())

    for i, batch in enumerate(tqdm(loader, desc=f'{time} train ({epoch})')):
        caption_data, video_ids, GTCAPS, starts, ends, categories, feature_stacks = batch
        meta_idx = caption_data.idx
        caption_idx = caption_data.caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        
        if 'i3d' in feature_names:
            if modality == 'video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow
            elif modality == 'audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks
            elif modality == 'subs_audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks, caption_data.subs
            else:
                raise Exception(f'it is not implemented for modality: {modality}')
        
        if 'audio_video' in modality:
            masks = mask(feature_stacks[0][:, :, 0], caption_idx, loader.dataset.pad_idx)
            if modality == 'subs_audio_video':
                subs_mask = mask(feature_stacks[-1], None, loader.dataset.pad_idx)
                masks = *masks, subs_mask
        else:
            masks = mask(feature_stacks[:, :, 0], caption_idx, loader.dataset.pad_idx)
        
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        if use_categories:
            pred = model(
                feature_stacks, caption_idx, masks, categories
            )
        else:
            pred = model(feature_stacks, caption_idx, masks)
        loss_iter = loss_compute(pred, caption_idx_y, n_tokens)
        loss_iter_norm = loss_iter / n_tokens
        losses.append(loss_iter_norm.item())

        if TBoard is not None:
            step_num = epoch * len(loader) + i
            TBoard.add_scalar('train/Loss_iter', loss_iter_norm.item(), step_num)
            TBoard.add_scalar('debug/lr', lr_scheduler.get_lr(), step_num)
    
    # we have already divided it
    loss_total_norm = np.sum(losses) / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', loss_total_norm, epoch)
            
def validation_next_word_loop(model, loader, decoder, loss_compute, lr_scheduler, 
                              epoch, max_len, videos_to_monitor, TBoard, 
                              modality, use_categories):
    model.eval()
    losses = []
        
    loader.dataset.update_iterator()
    
    time = strftime('%X', localtime())
    phase = loader.dataset.phase
    feature_names = loader.dataset.feature_names

    for i, batch in enumerate(tqdm(loader, desc=f'{time} {phase} ({epoch})')):
        caption_data, video_ids, GTCAPS, starts, ends, categories, feature_stacks = batch
        meta_idx = caption_data.idx
        caption_idx = caption_data.caption
        
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        
        if 'i3d' in feature_names:
            if modality == 'video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow
            elif modality == 'audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks
            elif modality == 'subs_audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks, caption_data.subs
            else:
                raise Exception(f'it is not implemented for modality: {modality}')
        
        if 'audio_video' in modality:
            masks = mask(feature_stacks[0][:, :, 0], caption_idx, loader.dataset.pad_idx)
            if modality == 'subs_audio_video':
                subs_mask = mask(feature_stacks[-1], None, loader.dataset.pad_idx)
                masks = *masks, subs_mask
        else:
            masks = mask(feature_stacks[:, :, 0], caption_idx, loader.dataset.pad_idx)
        
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            
        with torch.no_grad():
            if use_categories:
                pred = model(feature_stacks, caption_idx, masks, categories)
            else:
                pred = model(feature_stacks, caption_idx, masks)
            loss_iter = loss_compute.criterion(pred, caption_idx_y)
            loss_iter_norm = loss_iter / n_tokens
            losses.append(loss_iter_norm.item())
            
            if TBoard is not None:
                step_num = epoch * len(loader) + i
                TBoard.add_scalar(f'debug/{phase}_loss_iter', loss_iter_norm.item(), step_num)
                
    loss_total_norm = np.sum(losses) / len(loader)

    if TBoard is not None:
        TBoard.add_scalar(f'{phase}/Loss_epoch', loss_total_norm, epoch)
        
        if phase == 'val_1':
            if use_categories:
                text = predict_1by1_for_TBoard(
                    videos_to_monitor, loader, decoder, model, max_len, modality,
                    use_categories
                )
            else:
                text = predict_1by1_for_TBoard(
                    videos_to_monitor, loader, decoder, model, max_len, modality
                )
            TBoard.add_text(f'prediction_1by1_{phase}', text, epoch)
        
    return loss_total_norm

def validation_1by1_loop(model, loader, decoder, loss_compute, lr_scheduler, 
                         epoch, max_len, log_path, verbose,
                         reference_paths, tIoUs, max_prop_per_vid, TBoard, 
                         modality, use_categories):
    # todo: loss_compute, lr_scheduler can be removed but lr_scheduler can be used
    # if you have fancy scheduler
    start_timer = time()
    
    time_ = strftime('%X', localtime())
    
    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True, 
            'details': ''
        },
        'results': {}
    }
    model.eval()
    loader.dataset.update_iterator()
    
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase
    feature_names = loader.dataset.feature_names
    props_are_gt = loader.dataset.props_are_gt
    
    if props_are_gt:
        tqdm_title = f'{time_} 1-by-1 gt proposals ({epoch})'
        tIoUs = [0.5] # no need to wait
    else:
        tqdm_title = f'{time_} 1-by-1 predicted proposals ({epoch})'
        assert len(tIoUs) == 4
    
    for i, batch in enumerate(tqdm(loader, desc=tqdm_title)):
        caption_data, video_ids, GTCAPS, starts, ends, categories, feature_stacks = batch
        meta_idx = caption_data.idx
        caption_idx = caption_data.caption
        
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        
        if 'i3d' in feature_names:
            if modality == 'video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow
            elif modality == 'audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks
            elif modality == 'subs_audio_video':
                vid_feat_stacks_rgb, vid_feat_stacks_flow, aud_feat_stacks = feature_stacks
                feature_stacks = vid_feat_stacks_rgb + vid_feat_stacks_flow, aud_feat_stacks, caption_data.subs
            else:
                raise Exception(f'it is not implemented for modality: {modality}')
        
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        if use_categories:
            ints_stack = decoder(
                model, feature_stacks, max_len, start_idx, end_idx, pad_idx, 
                modality, categories)
        else:
            ints_stack = decoder(
                model, feature_stacks, max_len, start_idx, end_idx, pad_idx, 
                modality)
        ints_stack = ints_stack.cpu().numpy() # what happens here if I use only cpu?
        # transform integers into strings
        list_of_lists_with_strings = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack]

        ### FILTER PREDICTED TOKENS
        # initialize the list to fill it using indices instead of appending them
        list_of_lists_with_filtered_sentences = [None] * len(list_of_lists_with_strings)

        for b, strings in enumerate(list_of_lists_with_strings):
            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()
            # add the filtered sentense to the list
            list_of_lists_with_filtered_sentences[b] = sentence
            
        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sent in zip(video_ids, starts, ends, list_of_lists_with_filtered_sentences):
            segment = {
                'sentence': sent,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)

            else:
                predictions['results'][video_id] = [segment]
    
    if log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        if props_are_gt:
            save_filename = f'results_{phase}_e{epoch}.json'
        else:
            save_filename = f'results_val_pred_prop_e{epoch}_best.json'
        submission_path = os.path.join(log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(log_path, exist_ok=True)

        # if this is run with another loader and pretrained model
        # it substitutes the previous prediction
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        ## RUN THE EVALUATION
        # blocks the printing
        with HiddenPrints():
            val_metrics = calculate_metrics(reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose)

        ## WRITE TBOARD
        if (TBoard is not None) and (props_are_gt):
            # todo: add info that this metrics are calculated on val_1
            TBoard.add_scalar(f'{phase}/meteor', val_metrics['Average across tIoUs']['METEOR'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu4', val_metrics['Average across tIoUs']['Bleu_4'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu3', val_metrics['Average across tIoUs']['Bleu_3'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu2', val_metrics['Average across tIoUs']['Bleu_2'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu1', val_metrics['Average across tIoUs']['Bleu_1'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/rouge_l', val_metrics['Average across tIoUs']['ROUGE_L'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/cider', val_metrics['Average across tIoUs']['CIDEr'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/precision', val_metrics['Average across tIoUs']['Precision'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/recall', val_metrics['Average across tIoUs']['Recall'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/duration_of_1by1', (time() - start_timer) / 60, epoch)

        return val_metrics