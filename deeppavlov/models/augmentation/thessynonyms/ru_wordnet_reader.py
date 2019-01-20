import xml.etree.ElementTree as ET
import pymorphy2
from deeppavlov.core.data.utils import download_decompress
from pathlib import Path
from typing import List

from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)

#прикрутить морфотегер от Алексея
#он возвращает значения в universal... есть библиотека russian_tagsets которая переводит pymorphy тэги в universal...
#гиперонимы
class RuWordnet:
        
    def __init__(self, dir_path: str, with_source_token: bool=False):
         
        url = 'http://files.deeppavlov.ai/datasets/ruthes-lite2.tar.gz'
        url = "https://github.com/SultanovAR/___ruthes-lite2_download/archive/master.zip"   

        dir_path = Path(dir_path)
        required_files = ['text_entry.xml', 'synonyms.xml']
        if not dir_path.exists():
            dir_path.mkdir()
        
        if not all((dir_path/f).exists() for f in required_files):
            download_decompress(url, dir_path)

        self.with_source_token = with_source_token
        self.entry_root = ET.parse(dir_path / 'text_entry.xml').getroot()
        self.synonyms_root = ET.parse(dir_path / 'synonyms.xml').getroot()
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
        init_form = self._extract_morpho_requirements(init_form)
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

if __name__ == "__main__":
    print(RuWordnet('ruthes', True).get_synlist('адский'))
    #test 1: Downloading necessary data
    #import shutil
    #shutil.rmtree(Path('ruthes'), ignore_errors=True)
    #RuWordnet('ruthes')

    #test 2: if token == None, then algorithm returns None
    assert RuWordnet('ruthes').get_synlist(None) == None

    #test 3: it finds right synonyms and doesn't return source token
    #the reference list with synonyms was taken from http://www.labinform.ru/pub/ruthes/c/01/000/124951.htm
    assert RuWordnet('ruthes').get_synlist('адский') == ['кромешный']

    #test 4: it finds right synonyms and return source token
    #the reference list with synonyms was taken from http://www.labinform.ru/pub/ruthes/c/01/000/124951.htm
    assert set(RuWordnet('ruthes', True).get_synlist('адский')) == set(['адский', 'кромешный'])
    
    #test 5: checking synset for lemma "РАБ"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['НЕВОЛЬНИК', 'НЕВОЛЬНИЦА', 'НЕВОЛЬНИЧИЙ', 'РАБ', 'РАБСКИЙ', 'РАБЫНЯ']) == set(RuWordnet('ruthes')._find_synonyms('РАБ'))

    #test 6: checking synset for lemma "ЩЕБЕНЬ"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['ЩЕБЕНКА', 'ЩЕБЕНОЧНЫЙ', 'ЩЕБЕНЬ', 'ЩЕБНЕВОЙ', 'ЩЕБНЕВЫЙ']) == set(RuWordnet('ruthes')._find_synonyms('ЩЕБЕНЬ'))

    #test 7: checking result synset for "прочитана"
    # The right synset was taken from http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['дочитана', 'прочтена']) == set(RuWordnet('ruthes').get_synlist('прочитана'))