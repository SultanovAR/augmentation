from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.spelling_correction.electors.kenlm_elector import KenlmElector
from deeppavlov.core.data.utils import download_decompress

from nltk import sent_tokenize
from nltk import pos_tag_sents
from nltk.tokenize.moses import MosesDetokenizer
from itertools import repeat

from typing import List
from utils import RuWordNet, EnWordNet


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
    
    def _filter(self, token):
        if not(self.isalpha_only and token.isalpha()):
            return None
        if not(self.standard_cases_only and (self._check_case(token) == 'default')):
            return None
        return token

    def __init__(self, lang: str, standard_cases: dict=None, isalpha_only: bool=True,
                 standard_cases_only: bool=True, dir_path: str='ThesAug'):
        self.lang = lang
        self.isalpha_only = isalpha_only
        self.standard_cases_only = standard_cases_only
        self.tokenizer = NLTKMosesTokenizer()
        self.detokenizer = MosesDetokenizer()

        url_ru_model = "http://files.deeppavlov.ai/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz"
        url_en_model = "http://files.deeppavlov.ai/lang_models/en_wiki_no_punkt.arpa.binary.gz"

        if not dir_path.exists():
            dir_path.mkdir()

        if self.lang == 'rus':
            self.thesaurus = RuWordNet(dir_path)
            if not (dir_path / 'ru_wiyalen_no_punkt.arpa.binary').exists():
                download_decompress(url_ru_model, dir_path)
            self.lm = KenlmElector(dir_path / 'ru_wiyalen_no_punkt.arpa.binary')
        elif self.lang == 'eng':
            self.thesaurus = EnWordNet(dir_path)
            if not (dir_path / 'en_wiki_no_punkt.arpa.binary').exists():
                download_decompress(url_en_model, dir_path)
            self.lm = KenlmElector('en_wiki_no_punkt.arpa.binary')

        if standard_cases:
            self.standard_cases = standard_cases
        else:
            self.standard_cases = {'default': lambda x: x.lower(),\
                                    'upper': lambda x: x.upper(),\
                                    'capit': lambda x: x.capitalize(),\
                                    'lower': lambda x: x.lower()}

    #необходимо сделать чтобы была возможность подцепить deeppavlov NER 
    def _replace_by_synonyms(self, batch_text: List[str]):
        aug_batch_text = []
        for text in batch_text:
            sents = [self.tokenizer([sent]) for sent in sent_tokenize(text)][0]
            tagged_sents = pos_tag_sents(sents)
            for sent, tag_sent in zip(sents, tagged_sents):
                tag_sent = [tag[1] for tag in tag_sent]
                #save the standard cases, because the next processing will require tokens in lowercase 
                saved_cases = list(map(self._check_case, sent))
                #token filtering
                filtered_sent = list(map(self._filter, sent))
                #lowercasing
                filtered_sent = list(map(lambda x: x.lower() if x else x, filtered_sent))
                synset = [self.thesaurus.get_synset(token, pos_tag) for token, pos_tag in zip(filtered_sent, tag_sent)]
                synset = [syn if syn else source for syn, source in zip(synset, sent)]
                #transfor for kenlm_elector
                synset = list(map(lambda x: list(zip(repeat(1/len(x), len(x)), x)) if not isinstance(x, str) else [(1,x)], synset))
                augmented = self.lm([synset])[0]
                augmented = self.detokenizer.detokenize(augmented, return_str=True)
                aug_batch_text.append(augmented)
        return aug_batch_text

    def __call__():
        pass
