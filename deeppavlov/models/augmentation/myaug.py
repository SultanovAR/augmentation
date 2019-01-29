import pandas as pd
import sys

from deeppavlov.models.augmentation import ThesaurusAug
t = ThesaurusAug('rus', dir_path='./thesaug', freq_replace=1, with_source_token=True,\
                 beam_size=4, penalty_for_source_token=-12.45)

path_input = sys.argv[1]#'/Users/sultanovar/.deeppavlov/downloads/rusentiment/rusentiment_random_posts.csv'
path_output = sys.argv[2]

source_data = pd.read_csv(path_input).head()
source_labels = source_data['label']
source_text = list(source_data['text'])
aug_text = t._replace_by_synonyms(source_text)
aug_data = pd.concat([pd.Series(aug_text, name='text'), source_labels], axis=1)

aug_data.to_csv(path_output)