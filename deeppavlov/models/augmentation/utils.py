from itertools import repeat
from typing import List, Tuple
from collections import defaultdict
import pattern.en as en
from pattern.en import ADJECTIVE, NOUN, VERB, ADVERB
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import xml.etree.ElementTree as ET
import pymorphy2
from pathlib import Path
from typing import List

from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)

class EnWordNet:
    """
    It finds synonyms for each token in given list, for english language
    - 'None' token doesn't processing
    - it finds synonyms only for noun, adjective, verb, adverb
    - it finds synonyms exclude source token
    - for adjective, adjective satellite is included also in list of synonyms
    - all output synonyms are inflected to the source token form
    - all output words are in lowercase
    - errors in inflection are possible(pattern.en)

    Args:
        classical_pluralize: boolean, flags whether to use classical pluralize option in method pattern.en.pluralize
    Attributes:
        classical_pluralize: whether to use classical pluralize option in method pattern.en.pluralize
        wn_tag_to_pattern:   vocabulary between pos_tags in wordnet and pos_tags in pattern
    """

    def __init__(self, with_source_token: bool=False, classical_pluralize: bool=True):
        self.classical_pluralize = classical_pluralize
        self.wn_tag_to_pattern = {wn.VERB: VERB, wn.NOUN: NOUN, wn.ADJ: ADJECTIVE, wn.ADV: ADVERB}
        self.with_source_token = with_source_token
        #nltk.download('wordnet')   #Надо ли

    def _find_synonyms(self, lemma: str, pos_tag: str) -> List[str]:
        if pos_tag:    #if token pos in [noun, adj, adv, verb]
            synonyms = set()
            wn_synsets = wn.synsets(lemma, pos=pos_tag)
            if pos_tag == wn.ADJ:    #for adjective, it finds Adjective satellites, that connected with this adjective
                wn_synsets = wn_synsets.extend(wn.synsets(lemma, pos='s'))
            for synset in wn_synsets:
                if self.with_source_token:
                    syn = [synset_lemma.name() for synset_lemma in synset.lemmas()]
                else:
                    syn = [synset_lemma.name() for synset_lemma in synset.lemmas() if synset_lemma.name() != lemma]
                synonyms.update(syn)
            return list(synonyms)
        else:
            None

    def _lemmatize(self, token: str, pos_tag: str) -> Tuple[dict, str]:
        init_form, lemma = defaultdict(bool), token
        if pos_tag.startswith('N'):
            lemma = en.inflect.singularize(token)
            init_form.update({'pos': wn.NOUN, 'plur': pos_tag.endswith('S')})
        elif pos_tag.startswith('V'):
            lemma = en.lemma(token)
            init_form.update({'pos': wn.VERB, 'tense': en.tenses(token)[0]})
        elif pos_tag.startswith('J'):
            is_plur = en.inflect.pluralize(token, pos=ADJECTIVE, classical=self.classical_pluralize) == token
            init_form.update({'pos': wn.ADJ, 'comp': pos_tag.endswith('R'), 'supr': pos_tag.endswith('S'), 'plur': is_plur})
        elif pos_tag.startswith('R'):
            init_form.update({'pos': wn.ADV, 'comp': pos_tag.endswith('R'), 'supr': pos_tag.endswith('S')})
        return init_form, lemma
    
    def _inflect_to_init_form(self, token: str, init_form: str) -> str:
        token = token.split('_')    #it inflect only the first word in phrases
        if init_form['plur']:
            token[0] = en.inflect.pluralize(token[0], pos=self.wn_tag_to_pattern[init_form['pos']], classical=self.classical_pluralize)
        if init_form['tense']:
            token[0] = en.conjugate(token[0], init_form['tense'])
        if init_form['comp']:
            token[0] = en.inflect.comparative(token[0])
            if len(token[0].split()) == 2:  # if added 'more' or 'most' then need delete it
                token[0] = token[0].split()[1]
        if init_form['supr']:
            token[0] = en.inflect.superlative(token[0])
            if len(token[0].split()) == 2:  # if added 'more' or 'most' then need delete it
                token[0] = token[0].split()[1]
        return " ".join(token)
    
    def get_synlist(self, token: str, pos_tag: str) -> List[str]:
        """
        Generating list of synonyms for token
        - 'None' token doesn't processing
        - it finds synonyms only for noun, adjective, verb, adverb
        - it finds synonyms exclude source token, if param with_source_token == False
        - for adjective, adjective satellite is included also in list of synonyms
        - all output synonyms are inflected to the source token form
        - all output words are in lowercase
        - errors in inflection are possible(pattern.en)
        Args:
            token: for that token will be searched synonyms
            pos_tag: pos tags for token, in nltk.pos_tag format
        Return:
            List of synonyms, if no synonyms were found, return None
        """
        if token:
            init_form, lemma = self._lemmatize(token, pos_tag)
            synonyms = self._find_synonyms(lemma, init_form['pos'])
            if synonyms:
                synonyms = set(map(self._inflect_to_init_form, synonyms, repeat(init_form, len(synonyms))))
                return list(synonyms)
            else:
                return None
        else:
            return None

class RuWordNet:
    """
    asdfasdf
    """
    def __init__(self, dir_path: str, with_source_token: bool=False):
         
        url = 'http://files.deeppavlov.ai/ruthes-lite2.tar.gz' 

        dir_path = Path(dir_path)
        required_files = ['ruthes-lite2/text_entry.xml', 'ruthes-lite2/synonyms.xml']
        if not dir_path.exists():
            dir_path.mkdir()
        
        if not all((dir_path/f).exists() for f in required_files):
            download_decompress(url, dir_path)

        self.with_source_token = with_source_token
        self.entry_root = ET.parse(dir_path / 'ruthes-lite2' /'text_entry.xml').getroot()
        self.synonyms_root = ET.parse(dir_path / 'ruthes-lite2' / 'synonyms.xml').getroot()
        self.morph = pymorphy2.MorphAnalyzer()
    
    def _extract_morpho_requirements(self, tag: pymorphy2.analyzer.Parse):
        #Not all morphems are need for inflection
        if tag.POS in ["NOUN"]:
            keys = {"NOUN", "case", "number", "animacy", "person", "gender"}
        elif tag.POS in ["ADJF"]:
            keys = {"ADJF", "case", "gender", "number"}
        elif tag.POS == "ADJS":
            keys = {"ADJS", "gender", "number"}
        elif tag.POS == "VERB":
            keys = {"mood", "tense", "aspect", "person", "voice", "gender", "number"}
        elif tag.POS == "INFN":
            keys = {"INFN", "aspect"}
        elif tag.POS in ["PRTF"]:
            keys = {"PRTF", "case", "gender", "number", "voice", "aspect"}
        elif tag.POS == "PRTS":
            keys = {"PRTS", "gender", "number", "voice", "aspect"}
        elif tag.POS == "GRND":
            keys = {"GRND", "voice", "aspect"}
        elif tag.POS == "COMP":
            keys = {"COMP"}
        else:
            keys = {}
        values = {(getattr(tag, key) if key.islower() else key) for key in keys}
        return {x for x in values if x is not None}

    def _find_synonyms(self, lemma: str) -> List[str]:
        #parsing ruthes-lite-2 file
        lemma = lemma.upper()
        #1 
        entry_id_set = set(map(lambda x: x.get('id'), self.entry_root.findall(f"./entry[lemma='{lemma}']")))
        #2
        concept_id_set = set()
        for entry_id in entry_id_set:
            concept_id_set.update(set(map(lambda x: x.get('concept_id'), self.synonyms_root.findall(f"./entry_rel[@entry_id='{entry_id}']"))))
        #3
        syn_entry_id_set = set()
        for concept_id in concept_id_set:
            syn_entry_id_set.update(set(map(lambda x: x.get('entry_id'), self.synonyms_root.findall(f"./entry_rel[@concept_id='{concept_id}']"))))
        #4
        synlist = list()
        for syn_entry_id in syn_entry_id_set:
            synlist += list(map(lambda x: x.text, self.entry_root.findall(f"./entry[@id='{syn_entry_id}']/lemma")))
        return synlist
    
    def _filter(self, synlist: List[str], init_form: pymorphy2.analyzer.Parse, source_token: str) -> List[str]:
        #init_form = self._extract_morpho_requirements(init_form)
        init_form = init_form.grammemes
        filtered_synset = set()
        for syn in filter(lambda x: len(x.split()) == 1, synlist):
            inflected_syn = self.morph.parse(syn)[0].inflect(init_form)
            if inflected_syn:
                if not(self.with_source_token) and inflected_syn.word != source_token:
                    filtered_synset.update([inflected_syn.word])
                elif self.with_source_token:
                    filtered_synset.update([inflected_syn.word])
        return list(filtered_synset)

    def get_synlist(self, token: str, pos_tag: str=None) -> List[str]:
        if token:
            morphem = self.morph.parse(token)[0]
            synonyms = self._find_synonyms(morphem.normal_form)
            synonyms = self._filter(synonyms, morphem.tag, token)
            return synonyms

if __name__ == '__main__':
    #test 1: Finding the correct list of synonoms for noun,
    # the reference list was taken from the site
    # http://wordnetweb.princeton.edu/perl/webwn?s=eat&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=00000000
    assert set(EnWordNet().get_synlist('frog', 'NN')) == set(['toad', 'toad frog', 'anuran', 'batrachian', 'salientian', 'Gaul'])

    #test 2: if pos_tag not is in right format then return None
    assert EnWordNet().get_synlist('frog', 'ASDFfsdf') == None

    #test 3: inflect of noun
    assert set(EnWordNet().get_synlist('frogs', 'NNS')) == set(['toads', 'toads frog', 'anurans', 'batrachians', 'salientians', 'Gauls'])

    #test 4: process None token
    assert EnWordNet().get_synlist(None, 'NNS') == None

    #test 5: return None when synonyms don't found
    assert EnWordNet().get_synlist('Abrsdfl', 'NN') == None

    #test 6: inflect verb
    # the reference list was taken from the site
    # http://wordnetweb.princeton.edu/perl/webwn?s=eat&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=00000000
    assert set(EnWordNet().get_synlist('ate', 'VB')) == set(['feed', 'ate on', 'consumed',\
                                                             'ate up', 'used up', 'depleted',\
                                                             'exhausted', 'ran through', 'wiped out',\
                                                             'corroded', 'rusted'])

    #test 1: Downloading necessary data
    #import shutil
    #shutil.rmtree(Path('ruthes'), ignore_errors=True)
    #RuWordNet('ruthes')

    #test 2: if token == None, then algorithm returns None
    assert RuWordNet('ruthes').get_synlist(None) == None

    #test 3: it finds right synonyms and doesn't return source token
    #the reference list with synonyms was taken from http://www.labinform.ru/pub/ruthes/c/01/000/124951.htm
    assert RuWordNet('ruthes').get_synlist('адский') == ['кромешный']

    #test 4: it finds right synonyms and return source token
    #the reference list with synonyms was taken from http://www.labinform.ru/pub/ruthes/c/01/000/124951.htm
    assert set(RuWordNet('ruthes', True).get_synlist('адский')) == set(['адский', 'кромешный'])
    
    #test 5: checking synset for lemma "РАБ"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['НЕВОЛЬНИК', 'НЕВОЛЬНИЦА', 'НЕВОЛЬНИЧИЙ', 'РАБ', 'РАБСКИЙ', 'РАБЫНЯ']) == set(RuWordNet('ruthes')._find_synonyms('РАБ'))

    #test 6: checking synset for lemma "ЩЕБЕНЬ"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['ЩЕБЕНКА', 'ЩЕБЕНОЧНЫЙ', 'ЩЕБЕНЬ', 'ЩЕБНЕВОЙ', 'ЩЕБНЕВЫЙ']) == set(RuWordNet('ruthes')._find_synonyms('ЩЕБЕНЬ'))

    #test 7: checking result synset for "прочитана"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['дочитана', 'прочтена']) == set(RuWordNet('ruthes').get_synlist('прочитана'))