from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.spelling_correction.electors.kenlm_elector import KenlmElector
from deeppavlov.core.data.utils import download_decompress

from nltk import sent_tokenize
from nltk import pos_tag_sents
from nltk.tokenize.moses import MosesDetokenizer
from itertools import repeat
import random
from typing import List
from .utils import RuWordNet, EnWordNet
from pathlib import Path


@register('thesaurus_augmentation')
class ThesaurusAug(Component):
    """
    666
    """
    def _check_case(self, token):
        for case_type in self.standard_cases:
            if token == self.standard_cases[case_type](token):
                return case_type
        return 'default'
    
    def _filter(self, token, pos_tag):
        if not(self.isalpha_only and token.isalpha()):
            return None
        if self.standard_cases_only and (self._check_case(token) == 'default'):
            return None
        if not(pos_tag[0].lower() in self.available_pos_tags):
            return None
        if self.freq_replace < random.random():
            return None
        return token

    def __init__(self, lang: str, standard_cases: dict=None, isalpha_only: bool=True,
                 standard_cases_only: bool=True, dir_path: str='ThesAug',\
                 available_pos_tags: List[str]=['n', 'v', 'a', 'r'], freq_replace: float=0.7,\
                 with_source_token: bool=False, beam_size: int=6, penalty_for_source_token: float=0.9):
        self.lang = lang
        self.isalpha_only = isalpha_only
        self.standard_cases_only = standard_cases_only
        self.available_pos_tags = available_pos_tags
        self.freq_replace = freq_replace
        self.tokenizer = NLTKMosesTokenizer()
        self.detokenizer = MosesDetokenizer()
        self.with_source_token = with_source_token
        self.beam_size = beam_size
        self.penalty_for_source_token = penalty_for_source_token

        url_ru_model = "http://files.deeppavlov.ai/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz"
        url_en_model = "http://files.deeppavlov.ai/lang_models/en_wiki_no_punkt.arpa.binary.gz"

        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir()

        if self.lang == 'rus':
            self.thesaurus = RuWordNet(dir_path, with_source_token=False)
            if not (dir_path / 'ru_wiyalen_no_punkt.arpa.binary').exists():
                download_decompress(url_ru_model, dir_path)
            self.lm = KenlmElector(dir_path / 'ru_wiyalen_no_punkt.arpa.binary', beam_size=self.beam_size)
        elif self.lang == 'eng':
            self.thesaurus = EnWordNet(dir_path, with_source_token=False)
            if not (dir_path / 'en_wiki_no_punkt.arpa.binary').exists():
                download_decompress(url_en_model, dir_path)
            self.lm = KenlmElector('en_wiki_no_punkt.arpa.binary', beam_size=self.beam_size)

        if standard_cases:
            self.standard_cases = standard_cases
        else:
            self.standard_cases = { 'upper': lambda x: x.upper(),\
                                    'capit': lambda x: x.capitalize(),\
                                    'lower': lambda x: x.lower(),\
                                    'default': lambda x: x.lower()}
    
    def _transform_kenlm(self, source_token, synset):
        if isinstance(synset, str):
            return [(1,synset)]
        else:
            res =  [(self.penalty_for_source_token * 1/(len(synset)+1), source_token)]\
                     + list(zip(repeat(1/(len(synset)+1), len(synset)+1), synset))
            return res


    #необходимо сделать чтобы была возможность подцепить deeppavlov NER 
    def _replace_by_synonyms(self, batch_text: List[str]):
        aug_batch_text = []
        for text in batch_text:
            sents = [self.tokenizer([sent])[0] for sent in sent_tokenize(text)]
            tagged_sents = pos_tag_sents(sents)
            aug_sent = []
            for sent, tag_sent in zip(sents, tagged_sents):
                tag_sent = [tag[1] for tag in tag_sent] 
                saved_cases = list(map(self._check_case, sent))    #save the standard cases, because the next processing will require tokens in lowercase
                filtered_sent = list(map(self._filter, sent, tag_sent))
                filtered_sent = list(map(lambda x: x.lower() if x else x, filtered_sent))
                synset = [self.thesaurus.get_synlist(token, pos_tag) for token, pos_tag in zip(filtered_sent, tag_sent)]
                synset = [syn if syn else source for syn, source in zip(synset, sent)]
                if self.with_source_token:  #transfor for kenlm_elector
                    synset = list(map(self._transform_kenlm, sent, synset))
                else:
                    synset = list(map(lambda x: list(zip(repeat(1/len(x), len(x)), x)) if not isinstance(x, str) else [(1,x)], synset))
                augmented = self.lm([synset])[0]
                rest_cases = list(map(lambda case, aug: self.standard_cases[case](aug), saved_cases, augmented))
                rest_cases = self.detokenizer.detokenize(rest_cases, return_str=True)
                aug_sent.append(rest_cases)
            aug_batch_text.append(" ".join(aug_sent))
        return aug_batch_text

    def __call__(self):
        pass

if __name__ == "__main__":
    t = ThesaurusAug('rus', dir_path='./')
    ans = t._replace_by_synonyms(['Размеры плазмид варьируют от менее чем 1 тысячи пар оснований (п. о.) до 400—600 тысяч пар оснований.'])
    print(ans)