import os

import h5py
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchtext import data


def load_multimodal_features_from_h5(feat_h5_video, feat_h5_audio, feature_names_list, 
                                     video_id, start, end, duration, get_full_feat=False):
    supported_feature_names = {'i3d_features', 'c3d_features', 'vggish_features'}
    assert isinstance(feature_names_list, list)
    assert len(feature_names_list) > 0
    assert set(feature_names_list).issubset(supported_feature_names)

    if 'vggish_features' in feature_names_list:
        audio_stack = feat_h5_audio.get(f'{video_id}/vggish_features')

        # some videos doesn't have audio
        if audio_stack is None:
            print(f'audio_stack is None @ {video_id}')
            audio_stack = torch.empty((0, 128)).float()

        T_audio, D_audio = audio_stack.shape

    if 'i3d_features' in feature_names_list:
        video_stack_rgb = feat_h5_video.get(f'{video_id}/i3d_features/rgb')
        video_stack_flow = feat_h5_video.get(f'{video_id}/i3d_features/flow')
        
        assert video_stack_rgb.shape == video_stack_flow.shape
        T_video, D_video = video_stack_rgb.shape

        if T_video > T_audio:
            video_stack_rgb = video_stack_rgb[:T_audio, :]
            video_stack_flow = video_stack_flow[:T_audio, :]
            T = T_audio
        elif T_video < T_audio:
            audio_stack = audio_stack[:T_video, :]
            T = T_video
        else:
            # or T = T_audio
            T = T_video
        
        # at this point they should be the same
        assert audio_stack.shape[0] == video_stack_rgb.shape[0]
        
        # == if taking segments instead of full features
        if get_full_feat == False:
            start_quantile = start / duration
            end_quantile = end / duration
            start_idx = int(T * start_quantile)
            end_idx = int(T * end_quantile)
            # handles the case when a segment is too small
            if start_idx == end_idx:
                # if the small segment occurs in the end of a video
                # [T:T] -> [T-1:T] (T can be either T_video or T_audio)
                if start_idx == T:
                    start_idx -= 1
                # [T:T] -> [T:T+1]
                else:
                    end_idx += 1
            video_stack_rgb = video_stack_rgb[start_idx:end_idx, :]
            video_stack_flow = video_stack_flow[start_idx:end_idx, :]
            audio_stack = audio_stack[start_idx:end_idx, :]
            
        video_stack_rgb = torch.tensor(video_stack_rgb).float()
        video_stack_flow = torch.tensor(video_stack_flow).float()
        audio_stack = torch.tensor(audio_stack).float()
        
        return video_stack_rgb, video_stack_flow, audio_stack

    elif 'c3d_features' in feature_names_list:
        # c3d has only rgb
        video_stack_rgb = feat_h5_video.get(f'{video_id}/c3d_features')
        T_video, D_video = video_stack_rgb.shape

        if T_video > T_audio:
            video_stack_rgb = video_stack_rgb[:T_audio, :]
            T = T_audio
        elif T_video < T_audio:
            audio_stack = audio_stack[:T_video, :]
            T = T_video
        else:
        # or T = T_audio
            T = T_video
        
        # at this point they should be the same
        assert audio_stack.size(0) == video_stack_rgb.size(0)
                
        if get_full_feat == False:
            start_quantile = start / duration
            end_quantile = end / duration
            start_idx = int(T * start_quantile)
            end_idx = int(T * end_quantile)
            # handles the case when a segment is too small
            if start_idx == end_idx:
                # if the small segment occurs in the end of a video
                # [T:T] -> [T-1:T]
                if start_idx == T:
                    start_idx -= 1
                # [T:T] -> [T:T+1]
                else:
                    end_idx += 1
            video_stack_rgb = video_stack_rgb[start_idx:end_idx, :]
            audio_stack = audio_stack[start_idx:end_idx, :]
        video_stack_rgb = torch.tensor(video_stack_rgb).float()
        audio_stack = torch.tensor(audio_stack).float()
        
        return video_stack_rgb, audio_stack
    
    else:
        raise Exception(f'Inspect: "{feature_names_list}"')

def filter_features(tensor, average_split, remove_overlap=True, feat_size=500):
    split_size = 4 # c3d: 16 x 4 = 64
    
    if len(tensor) == 0:
        return tensor
    
    if remove_overlap:
        # remove overlap 2 = 16/8 = stacksize/stepsize
        tensor = tensor[::2, :]
    
    if average_split:
        # make 1 feature out of 4: 4 = 64/16 = i3d_stacksize/c3d_stacksize 
        tensor = [split.mean(dim=0) for split in torch.split(tensor, split_size)]
        tensor = torch.stack(tensor)
        
    return tensor

def caption_iterator(start_token, end_token, pad_token, train_meta_path, val_1_meta_path,
                     val_2_meta_path, min_freq, batch_size, device, phase, use_categories, 
                     use_subs):
    spacy_en = spacy.load('en')
    print(f'Preparing dataset for {phase}')
    
    def tokenize_en(txt):
        return [token.text for token in spacy_en.tokenizer(txt)]
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', init_token=start_token, 
        eos_token=end_token, pad_token=pad_token, lower=True, 
        batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    if use_categories:
        # preprocessing: if there is no category replace with -1 (unique number)
        CATEGORY = data.Field(
            sequential=False, use_vocab=False, batch_first=True, 
            preprocessing=data.Pipeline(lambda x: -1 if len(x) == 0 else int(float(x)))
        )
        # filter the dataset if the a category is missing (31 -> 41 (count = 1 :()))
        filter_pred = lambda x: vars(x)['category_32'] != -1 and vars(x)['category_32'] != 31
    else:
        CATEGORY = None
        filter_pred = None
    
    if use_subs:
        SUBS = data.ReversibleField(
            tokenize='spacy', init_token=start_token, 
            eos_token=end_token, pad_token=pad_token, lower=True, 
            batch_first=True
        )
    else:
        SUBS = None
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('category_32', CATEGORY),
        ('subs', SUBS),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=train_meta_path, format='tsv', skip_header=True, fields=fields,
        filter_pred=filter_pred
    )
    CAPTION.build_vocab(dataset.caption, min_freq=min_freq)
    train_vocab = CAPTION.vocab
    
    train_subs_vocab = None
    if use_subs:
        SUBS.build_vocab(dataset.subs, min_freq=min_freq)
        train_subs_vocab = SUBS.vocab
        
    if phase == 'val_1':
        dataset = data.TabularDataset(
            path=val_1_meta_path, format='tsv', skip_header=True, fields=fields,
            filter_pred=filter_pred
        )
    elif phase == 'val_2':
        dataset = data.TabularDataset(
            path=val_2_meta_path, format='tsv', skip_header=True, fields=fields, 
            filter_pred=filter_pred
        )
    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(x.caption))
    sort_key = lambda x: 0 #len(x.caption)
    datasetloader = data.BucketIterator(
        dataset, batch_size, sort_key=sort_key, device=device, repeat=False, shuffle=True
    )
    return train_vocab, train_subs_vocab, datasetloader

class AudioVideoFeaturesDataset(Dataset):
    
    def __init__(self, video_features_path, video_feature_name, 
                 audio_features_path, audio_feature_name, 
                 meta_path, filter_video_feats, average_video_feats, 
                 filter_audio_feats, average_audio_feats, device, pad_idx, 
                 get_full_feat):
        self.video_features_path = video_features_path
        self.video_feature_name = f'{video_feature_name}_features'
        self.audio_features_path = audio_features_path
        self.audio_feature_name = f'{audio_feature_name}_features'
        self.feature_names_list = [self.video_feature_name, self.audio_feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.feat_h5_video = h5py.File(video_features_path, 'r')
        self.feat_h5_audio = h5py.File(audio_features_path, 'r')
        self.filter_video_feats = filter_video_feats
        self.average_video_feats = average_video_feats
        self.filter_audio_feats = filter_audio_feats
        self.average_audio_feats = average_audio_feats
        self.get_full_feat = get_full_feat
        
        if self.video_feature_name == 'i3d_features':
            self.video_feature_size = 1024
        elif self.video_feature_name == 'c3d_features':
            self.video_feature_size = 500
        else:
            raise Exception(f'Inspect: "{self.video_feature_name}"')
            
        if self.audio_feature_name == 'vggish_features':
            self.audio_feature_size = 128
        else:
            raise Exception(f'Inspect: "{self.audio_feature_name}"')
            
    def getitem_1_stream_video(self, indices):
        video_ids, captions, starts, ends, categories = [], [], [], [], []
        vid_stacks, aud_stacks = [], []

        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, category, _, _, _ = self.dataset.iloc[idx]
            
            # load the features from the hdf5
            vid_stack, aud_stack = load_multimodal_features_from_h5(
                self.feat_h5_video, self.feat_h5_audio, self.feature_names_list, 
                video_id, start, end, duration, self.get_full_feat
            )
            
            assert vid_stack.shape[1] == self.video_feature_size
            assert aud_stack.shape[1] == self.audio_feature_size
            
            # sometimes vid_stack and aud_stack are empty after the filtering. 
            # we replace it with noise.
            # since they are the same, there is no need to check len(aud_stack) == 0 
            if len(vid_stack) == 0:
                print(f'len(vid_stack) == 0 and len(aud_stack) == 0 @: {video_id}')
                vid_stack = torch.rand(1, self.video_feature_size, device=self.device)
                aud_stack = torch.rand(1, self.audio_feature_size, device=self.device)
            else:
                if self.filter_video_feats:
                    vid_stack = filter_features(vid_stack, self.average_video_feats)
                if self.filter_audio_feats:
                    aud_stack = filter_features(aud_stack, self.average_audio_feats)
            
            vid_stack = vid_stack.to(self.device)
            aud_stack = aud_stack.to(self.device)
            
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            categories.append(category)
            vid_stacks.append(vid_stack)
            aud_stacks.append(aud_stack)
            
        vid_stacks = pad_sequence(vid_stacks, batch_first=True, padding_value=self.pad_idx)
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)
                
        starts = torch.tensor(starts, device=self.device).unsqueeze(1)
        ends = torch.tensor(ends, device=self.device).unsqueeze(1)
        categories = torch.tensor(categories, device=self.device).unsqueeze(1)
        
        return video_ids, captions, starts, ends, categories, (vid_stacks, aud_stacks)
    
    def getitem_2_stream_video(self, indices):
        video_ids, captions, starts, ends, categories = [], [], [], [], []
        vid_stacks_rgb, vid_stacks_flow, aud_stacks = [], [], []
        
        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, category, _, _, _ = self.dataset.iloc[idx]
            
            # load the features from the hdf5
            vid_stack_rgb, vid_stack_flow, aud_stack = load_multimodal_features_from_h5(
                self.feat_h5_video, self.feat_h5_audio, self.feature_names_list, 
                video_id, start, end, duration, self.get_full_feat
            )
            
            assert vid_stack_rgb.shape[1] == self.video_feature_size
            assert vid_stack_flow.shape[1] == self.video_feature_size
            assert aud_stack.shape[1] == self.audio_feature_size
            
            # sometimes vid_stack and aud_stack are empty after the filtering. 
            # we replace it with noise.
            # since they are the same, there is no need to check len(aud_stack) == 0
            if len(vid_stack_rgb) == 0:
                print(f'len(vid_stack) == 0 and len(aud_stack) == 0 @: {video_id}')
                vid_stack_rgb = torch.rand(1, self.video_feature_size, device=self.device)
                vid_stack_flow = torch.rand(1, self.video_feature_size, device=self.device)
                aud_stack = torch.rand(1, self.audio_feature_size, device=self.device)
            else:
                if self.filter_video_feats:
                    vid_stack_rgb = filter_features(vid_stack_rgb, self.average_video_feats)
                    vid_stack_flow = filter_features(vid_stack_flow, self.average_video_feats)
                if self.filter_audio_feats:
                    aud_stack = filter_features(aud_stack, self.average_audio_feats)
            
            vid_stack_rgb = vid_stack_rgb.to(self.device)
            vid_stack_flow = vid_stack_flow.to(self.device)
            aud_stack = aud_stack.to(self.device)
            
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            categories.append(category)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            aud_stacks.append(aud_stack)
            
        # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)
                
        starts = torch.tensor(starts, device=self.device).unsqueeze(1)
        ends = torch.tensor(ends, device=self.device).unsqueeze(1)
        categories = torch.tensor(categories, device=self.device).unsqueeze(1)
        
        return video_ids, captions, starts, ends, categories, (vid_stacks_rgb, vid_stacks_flow, aud_stacks)
        
    def __getitem__(self, indices):
        
        if 'i3d_features' in self.feature_names_list:
            return self.getitem_2_stream_video(indices)
        elif 'c3d_features' in self.feature_names_list:
            return self.getitem_1_stream_video(indices)
        else:
            raise Exception(f'Inspect: "{self.feature_names_list}"')

    def __len__(self):
        return len(self.dataset)

class ActivityNetCaptionsIteratorDataset(Dataset):
    
    def __init__(self, start_token, end_token, pad_token, min_freq, batch_size,
                 video_features_path, video_feature_name, 
                 filter_video_feats, average_video_feats, 
                 audio_features_path, audio_feature_name, 
                 filter_audio_feats, average_audio_feats, train_meta_path, 
                 val_1_meta_path, val_2_meta_path, device, phase, modality,
                 use_categories, props_are_gt, get_full_feat, show_i3d_preds=None):
        '''
            For the doc see the __getitem__.
        '''
        
        self.device = device
        self.phase = phase
        self.batch_size = batch_size
        self.video_features_path = video_features_path
        self.video_feature_name = video_feature_name
        self.audio_features_path = audio_features_path
        self.audio_feature_name = audio_feature_name
        self.feature_names = f'{video_feature_name}_{audio_feature_name}'
        
        if modality == 'subs_audio_video':
            self.use_subs = True
        else:
            self.use_subs = False
        
        # caption dataset *iterator*
        self.train_vocab, self.train_subs_vocab, self.caption_loader = caption_iterator(
            start_token, end_token, pad_token, train_meta_path, val_1_meta_path, 
            val_2_meta_path, min_freq, batch_size, device, phase, use_categories,
            self.use_subs
        )
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[pad_token]
        self.start_idx = self.train_vocab.stoi[start_token]
        self.end_idx = self.train_vocab.stoi[end_token]
        
        if phase == 'train':
            meta_path = train_meta_path
        elif phase == 'val_1':
            meta_path = val_1_meta_path
        elif phase == 'val_2':
            meta_path = val_2_meta_path
        else:
            assert True == False, f'handle the new phase {phase}'
            
        self.get_full_feat = get_full_feat
        
        if modality == 'audio_video' or modality == 'subs_audio_video':
            self.features_dataset = AudioVideoFeaturesDataset(
               video_features_path, video_feature_name, 
               audio_features_path, audio_feature_name, 
               meta_path, filter_video_feats, average_video_feats, 
               filter_audio_feats, average_audio_feats, device, 
               self.pad_idx, self.get_full_feat
            )
            if modality == 'subs_audio_video':
                self.subs_voc_size = len(self.train_subs_vocab)
        else:
            raise Exception(f'it is not implemented for modality: {modality}')
            
        self.modality = modality
        self.use_categories = use_categories
        self.props_are_gt = props_are_gt
        
        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)
        
    def __getitem__(self, dataset_index):
        caption_data = next(self.caption_loader_iter)
        
        # a note about "*": 1, 2, *('a', 'b', 'c') -> 1, 2, 'a', 'b', 'c'
        to_return = caption_data, *self.features_dataset[caption_data.idx]
        return to_return

    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''
            This should be called after every epoch
        '''
        self.caption_loader_iter = iter(self.caption_loader)
        
    def dont_collate(self, batch):
        return batch[0]
