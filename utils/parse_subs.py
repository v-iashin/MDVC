import os
import re
import pandas as pd

def parse_timestamp(timestamp):
    '''
        Extracts start and end points of a subtitles
        '00:00:01,320 --> 00:00:19,609'
    '''
    start, end = timestamp.replace(',', '.').split(' --> ')
    s_hours, s_mins, s_secs = start.split(':')
    e_hours, e_mins, e_secs = end.split(':')
    s_secs = int(s_hours)*3600 + int(s_mins)*60 + round(float(s_secs), 2)
    e_secs = int(e_hours)*3600 + int(e_mins)*60 + round(float(e_secs), 2)
    
    # floats
    return s_secs, e_secs

def clean_text(text):
    '''
    cleans things like:
        <font color="#E5E5E5">you</font><font color="#CCCCCC"> just better</font> you just better
        [Music]
        [Applause]
    '''
    text = re.sub('<[^>]*>', ' ', text).strip()
    text = re.sub('\s{2,}', ' ', text)
    text = text.replace('[Music]', '')
    text = text.replace('[Applause]', '')
    
    # str
    return text

def parse_sub(in_stream):
    '''
        Extracts start, end, and subtitle from a two lines given the input stream open()
    '''
    # read new lines that contain the content of a sub
    num = in_stream.readline().replace('\n', '')
    
    # end file
    if len(num) == 0:
        return None, None, None
    
    timestamp = in_stream.readline().replace('\n', '')
    assert len(timestamp) != 0
    text = in_stream.readline().replace('\n', '')

    # moving pointer over the empty line
    in_stream.readline()
    
    # extract start and end times
    start_secs, end_secs = parse_timestamp(timestamp)
    # clean the content of a sub
    text = clean_text(text)
    
    # floats and str
    return start_secs, end_secs, text

def parse_sub_file(path):
    '''
        Parses a subtitle file.
    '''
    starts, ends, texts = [], [], []
    in_stream = open(path, 'r')
    
    # while the end of the file has been reached
    while in_stream:
        start_secs, end_secs, text = parse_sub(in_stream)
            
        if (start_secs is None) or (end_secs is None) or (text is None):
            break
        else:
            starts.append(start_secs)
            ends.append(end_secs)
            texts.append(text)
    
    # sanity check
    line_number = len(open(path, 'r').readlines())
    if (line_number - len(texts) * 4) > 1:
        print(path, line_number, len(texts) * 4)
    
    # lists
    return starts, ends, texts

def add_adjusted_end_2_df(subs_dataframe):
    '''
        Given a pandas dataframe, adjusts the start and end points to address the following problem:

        YouTube displays the previous speech segment, as well as the new one, appears when
        somebody is speaking. When the current line is finished, it replaces the previous one
        while the new one start to appear on the screen and so on. Therefore, the starting 
        point is quite accurate when the ending point is not. Considering the fact that the 
        previous speech segment is ended by the start of the next one, we may adjust the 
        ending point to be the start of the next segment within one video.
    '''
    subs_dataframe['video_id_next'] = subs_dataframe['video_id'].shift(periods=-1)
    subs_dataframe['start_next'] = subs_dataframe['start'].shift(periods=-1)
    subs_dataframe['end_next'] = subs_dataframe['end'].shift(periods=-1)
    
    # defining it here to use in in dataframe.apply instead of a lambda funcion
    def adjust_end_time(row):
        if row['video_id_next'] == row['video_id']:
            return min(row['end'], row['start_next'])
        else:
            return row['end']

    subs_dataframe['end_adj'] = subs_dataframe.apply(adjust_end_time, axis=1)
    # filter columns that end with '_next' (temp columns)
    subs_dataframe = subs_dataframe.filter(regex='.+(?<!_next)$')
    
    return subs_dataframe

def filter_dataframe(dataframe):
    '''
        Some sanity check filtering: start ponint is too far
        or if sub is an empty string
    '''
    dataframe = dataframe[dataframe['start'] < 5000].reset_index(drop=True)
    dataframe = dataframe[dataframe['sub'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    return dataframe

def subtitles_dataframe(subs_folders, save_path=None):
    '''
        creates a pd.DataFrame object and saves .csv
    '''
    
    video_ids_acc = []
    starts_acc = []
    ends_acc = []
    subs_acc = []
    comments_acc = []

    # repeats the same procedure for each folder with subs (en, translated, other)
    for subs_folder in subs_folders:
        # extracts the folder name as the comment column
        comment = os.path.basename(os.path.normpath(subs_folder))
        for i, filename in enumerate(sorted(os.listdir(subs_folder))):
            filename_path = os.path.join(subs_folder, filename)
            starts, ends, subs = parse_sub_file(filename_path)
            video_id = f'v_{filename[:11]}'
            video_ids_acc += [video_id] * len(starts)
            starts_acc += starts
            ends_acc += ends
            subs_acc += subs
            comments_acc += [comment] * len(starts)
        
    dataframe = pd.DataFrame({
        'video_id': video_ids_acc,
        'sub': subs_acc,
        'start': starts_acc,
        'end': ends_acc,
        'comment': comments_acc
    })
    
    dataframe = add_adjusted_end_2_df(dataframe)
    print(f'Dataset size before filtering: {dataframe.shape}')
    dataframe = filter_dataframe(dataframe)
    print(f'Dataset size after filtering: {dataframe.shape}')
    print(f'save_path: {save_path}')
    if save_path is not None:
        dataframe.to_csv(save_path, index=None, sep='\t')
    return dataframe


if __name__ == "__main__":
    '''
        (mdvc) $ python ./utils/parse_subs.py
    '''
    # make sure to unzip the subs.zip
    # we are using only `en` folder but you can play with other ones
    # we tried with `en` + `translated` but it didn't improve the results
    subs_folders = [f'./data/subs/asr_en/']
    save_path = f'./data/asr_en.csv'
    subs_dataframe = subtitles_dataframe(subs_folders, save_path)
    print(subs_dataframe.tail())

    # check if both files are the same upto sorting: os.listdir(subs_folder) doesn't gurantee to return
    # a sorted list. Now code ensures the list of filenames to be lexicographically sorted.
    # old = pd.read_csv('./data/asr_en_new.csv', sep='\t').values.tolist()
    # new = pd.read_csv('./data/asr_en.csv', sep='\t').values.tolist()
    # print(len(old), len(new))
    # from tqdm import tqdm
    # for line in tqdm(old):
    #     assert line in new
    # for line in tqdm(new):
    #     assert line in old
