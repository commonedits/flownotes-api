import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict as cmu
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords 
from topia.termextract import extract
extractor = extract.TermExtractor()
import pattern.en as ptn

import urllib, json
import requests
from re import sub, split
import random
from random import randrange, shuffle

stop = map(lambda string: string.encode('ascii','ignore'), stopwords.words('english'))

Phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
Consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
Vowels =  ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

num_phonemes = len(Phonemes)
max_synonyms = 100

using_topia_tagger = False

prondict = cmu.dict() 
lemmatizer = WordNetLemmatizer()




def extract_words(sentence):
    # tokenized sentence. returns a list of [word, part of speech, word_singular]
    if(using_topia_tagger):
        tok = extractor.tagger(sentence)
        if not tok:
            return []
        # hack for fixing the bug which sometimes makes extractor's first word of the senten lower case
        first = sentence.strip()[:1] # first letter
        if (first.upper() == first):
            tok[0][0] = tok[0][0][:1].upper() + tok[0][0][1:]
        else:
            tok[0][0] = tok[0][0][:1] + tok[0][0][1:]
    else:
        # use nltk's tagger maxent_treebank_pos_tagger
        text = word_tokenize(sentence)
        tok = pos_tag(text)
    """
    # pattern's tagger:
    ptn.tag(text)
    """
        
    #if DEBUG:
        #print tok 
        
    # nothing to extract
    if len(tok) == 0:
        return []
    
    
    formatted = map(lambda w: {"word_original": w[0], "word_new": w[0], "ignore": False, "pos": w[1], "word_lower": justlowerletters(w[0])}, tok)
    

    return formatted
    #sentence = justlowerletters(sentence)
    #return sentence.split(" ")
    
# given string, returns ascii string, lowercase
def justlowerletters(string):
    string = string.encode('ascii','ignore')
    #string = str(string)
    return str.lower(string)
    #return sub(r'\s+',' ',sub(r'[^a-z\s]', '', str.lower(string)))


def phonetilicious_score(loWO, flatten=False, normalize=False):
    if not loWO:
        return 0
    
    """ CHANGE THIS, MAKE IT BETTER """
    
    phonevectors = map(lambda WO: WO["phonevector"], loWO)
    
    
    if flatten:
        # set each word's phoneme count to 0 or 1.
        phonevectors = map(lambda v: map(lambda p: 0 if p == 0 else 1, v),  phonevectors)
    
    # sum up phonevectors for each word
    vector_sum = reduce(lambda v,w: v + w, phonevectors)
    
    score = 0.0

    phone_count = 0
    for n in vector_sum:
        if n > 0:
            score += n * n * n  # recurring phones get cubed points
            phone_count += n
    
    if normalize:
        score /= phone_count
        
    # count repeated words
    # no, this isn't the place to do this, because by now we've already maximized one example for each phoneme
    # do this earlier
    
    words = map(lambda WO: WO["word_lower"], loWO)
    word_doubles = 0
    for word in words:
        extras = words.count(word) - 1
        if extras > 0:
            word_doubles += extras
    score /= (0.2 * (word_doubles+1)) #severely penalized for extra words
    
    return score 
    
"""
Word Object looks like:
,
WO = {    'word_original': 'Truly',
          'word_lower': 'truly'
        'pos': 'RB',
        'ignore': true,
        'phonevector': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])}
"""
# given a phone ph, and list of Word Objects, returns the Word Object with max occurance of ph
# try to avoid words on blacklist (word_lower)
def maxphone(ph, loWO, blacklist=[]):
    if blacklist: 
        new_loWO = [WO for WO in loWO if WO["word_lower"] not in blacklist]
        if len(new_loWO) == 0:
            new_loWO = loWO
    else:
        new_loWO = loWO
    
    result = max(new_loWO, key = lambda x: x['phonevector'][Phonemes.index(ph)])
    
    return result

        

# format new word with the capitalization of the old word
def format_capitalization(new_word, old_word):
    if old_word.capitalize() == old_word:
        return new_word.capitalize()
    elif old_word.upper() == old_word:
        return new_word.upper()
    elif old_word.lower() == old_word:
        return new_word.lower()
    else:
        # if it's none of these, then it's something weirder
        return new_word

def pos_topia2datamuse(pos):
    x = pos[:1]
    if x == "N":
        return "n"
    elif x == "V":
        return "v"
    elif x == "J":
        return "adj"
    elif x == "R":
        return "adv"
    else:
        return x.lower()
    
    
def pos_topia2wordnet(pos):
    x = pos[:1]
    if x == "N":
        return "n"
    elif x == "V":
        return "v"
    elif x == "J":
        return "a"
    elif x == "R":
        return "r"
    else:
        return x.lower()
    
def algo(text, 
         alliteration_only = False, 
         number_of_examples = 4, 
         force_phoneme = [], 
         flatten = False, 
         split_by = "sentences", # "sentences" or "punctuation"
         normalize = False, 
         try_to_avoid_dups = False, 
         word_additions = [],
         swap_probability = 1.0,
         hyper_profanity = False,
         ignore_numbers = True,
         debug = False):
    
    DEBUG = debug
    
    out = []
    
    if split_by == "punctuation":
		sentences = re.split(r'( *[\.\?\;\,\:\—\n!][\'"\)\]]* *)', text)
    elif split_by == "lines":
        sentences = re.split(r'(\n)', text)
    else:
        sentences = sent_tokenize(text)
    
    if hyper_profanity:
        word_additions += PROFANITY
    
    if DEBUG:
        print sentences
        
    for sentence in sentences: 
        
        # should be tokenized with part of speech, and commonness
        words = extract_words(sentence)
        
        if not words:
            continue #empty sentence
        

        # which words are we keeping?
        
        # just nouns, verbs, adjectives, adverbs
        pos = ["N","V","J","R"]
        
        for w in words:
            w["ignore"] = (w['word_lower'] in stop) or not (w['pos'][:1] in pos)
            w["phonevector"] = phonevector(phones(w["word_lower"]), alliteration_only)
        
        #if DEBUG: 
            #print words
        if DEBUG:
            print words
        
        # for each word, find a list of related Word Objects
        lolorWO = map(lambda w: 
                      [w] if w["ignore"] else
                      map(lambda rw: {
                            "word_lower": rw, 
                            "word_original": w['word_original'],
                            "word_new": format_capitalization(rw, w['word_original']),
                            "pos": w['pos'],
                            "phonevector": phonevector(phones(rw), alliteration_only)
                        }, possible_words(w["word_lower"],w['pos'], word_additions, swap_probability, ignore_numbers)),
                    words) 
        
        if DEBUG: 
            print len(lolorWO)
            
        def generate_sentence_with_maxphone(p,lolorWO):
            sentence = []
            blacklist = []
            for loWO in lolorWO:
                # This needs a shuffle, else it will always pick the same max
                shuffle(loWO)
                mp = maxphone(p, loWO, blacklist)
                if try_to_avoid_dups:
                    blacklist.append(mp["word_lower"])
                sentence.append(mp)
            return sentence
            #sentence = map(lambda loWO: maxphone(p, loWO), lolorWO)
            
        
        if len(force_phoneme) == 0:
            phonemes_to_maximize = Phonemes
        else:
            phonemes_to_maximize = force_phoneme
        loS = map(lambda p: generate_sentence_with_maxphone(p, lolorWO), phonemes_to_maximize)
        
        #if DEBUG:
            #print loS
        
        
        
        
        sentences_with_score = map(lambda s: {'sentence': sentencify(s, sentence), 'score': phonetilicious_score(s, flatten, normalize)}, loS)
        
        #indexes = unique(sentences_with_score, return_index=True)[1]
        #sentences_unique = [sentences_with_score[index] for index in sorted(indexes)]
        
        # unique array, still sorted by score. 
        # http://stackoverflow.com/questions/12926898/numpy-unique-without-sort
        
        sentences_unique = {v['sentence']:v for v in sentences_with_score}.values()
        
        best_sentences = sorted(sentences_unique, key=lambda s: s['score'], reverse=True)
        
        if DEBUG:
            for s in best_sentences:
                print s
                
        # trim down number of examples
        best_sentences = best_sentences[:number_of_examples]
                
        if DEBUG:
            print "\nBEST SENTENCE:"
            print best_sentences[0]
            print "\n"
        
        for s in best_sentences:
            out.append(s['sentence'])
        
    return out

# given a list of Word Objects, returns sentence
def sentencify(loWO, original_sentence):
    
    """
    # remove unnecessary space around punctuation
    i = 0
    while i<len(words):
        if i>0 and (words[i] in PUNCTUATION):
            words[i-1] = words[i-1] + words[i]
            words.pop(i)
        else:
            i+=1
    """

    
    # this is wrong, 
    # because if our replacements are [B->C, C->D] 
    # this will incorrectly go  A B C => A C C => A D C  
    # instead of                A B C => A C D
    x = 0
    for WO in loWO:
        original_sentence = original_sentence[:x] + original_sentence[x:].replace(WO['word_original'], WO['word_new'], 1)
        x += len(WO['word_new'])
    
    words = original_sentence.split(" ")
        
    # fix an; "a egg"  -> "an egg"
    i = 0
    for (i,word) in enumerate(words):
        if (i+1) < len(words):
            vowel_word = words[i+1][:1].lower() in ["a","e","i","o","u"]
            if word.lower() == "a" and vowel_word:
                words[i] = format_capitalization("an",word)
            elif word.lower() == "an" and not vowel_word:
                words[i] = format_capitalization("a",word)
    
    return " ".join(words)
   
# given a word, return list of phones
def phones(word):
    if word in prondict:
        return prondict[word][0]
    else: # word is not in cmudict, return empty list
        return []
    
# given a list of phones, return phone vector.
def phonevector(lop, alliteration_only):
    vector = [0]*num_phonemes

    for p in lop:
        #if(p[2:]=="0"):
            #continue # ignore unstressed vowels
            
        p = p[:2] # treat primary and secondary stress the same
    
        vector[Phonemes.index(p)] += 1 #= 1 to flatten phonemes
        if alliteration_only: 
            # only look at the first letter
            break
        
    return list(vector)



def conjugate_noun(noun, pos):
    if pos=="NNS" or pos =="NNPS":
        return str(ptn.pluralize(noun))
    elif pos=="NN" or pos =="NNP":
        return str(ptn.singularize(noun))
    else:
        return noun
    
# conjugate verb into new pos
def conjugate_verb(verb, pos):
    if pos[:1]=="V":
        # verb which isn't be, do, have, or will/would/can/could
        # we need to conjugate
        #print verb, pos
        #print ptn.tenses(verb)
        #print ptn.lexeme(verb)
        #print ptn.lemma(verb)
        
        if pos[2:3]=="B":
            conj = ptn.conjugate(verb, tense = "infinitive")
        elif pos[2:3]=="D":
            conj = ptn.conjugate(verb, tense = "past")
        elif pos[2:3]=="G":
            conj = ptn.conjugate(verb, tense = "present", aspect = "progressive")
        elif pos[2:3]=="I":
            conj = ptn.conjugate(verb, tense = "infinitive")
        elif pos[2:3]=="N":
            conj = ptn.conjugate(verb, tense = "past", aspect="progressive")
        elif pos[2:3]=="Z":
            conj = ptn.conjugate(verb, tense = "present", person = 3, number = "singular")
        else:
            conj = verb
    return str(conj)




# given a word, return list of related words
# pos = part of speech
# word_additions = extra words to choose from (ex: profanity)
# addition_probably = probably that we add the word_additions to the new word list (0 to 1)
def possible_words(word, pos, word_additions, swap_probability=1.0, ignore_numbers = True):
    
    if(swap_probability <= random.random()):
        return [word]
    
    if ignore_numbers and (word.strip().isdigit()):
        return [word]
    
    pos_datamuse = pos_topia2datamuse(pos)
   
    """
    pos_wordnet = pos_topia2wordnet(pos)
    
    # synonyms
    #lemma = lemmatizer.lemmatize(word, pos) 
    syn_sets = wn.synsets(word,pos_wordnet)

    all_syn_sets = syn_sets
    
    #add hypernyms
    for syn_set in syn_sets:
        #all_syn_sets += syn_set.hypernyms()
        all_syn_sets += syn_set.hyponyms()
        
    words = []

    # TODO: pick synsets based on part of speech
    for syn_set in all_syn_sets:
        for w in syn_set.lemma_names():
            if w not in words:
                if w and ("_" not in w) and (w is not word):  # for now, ignore multi-word synonyms
                    words.append(w)
            elif len(words) >= max_synonyms:    
                break
        if len(words) >= max_synonyms:    
            break
    """
    
    urls = ["https://api.datamuse.com/words?ml=%s&max=100&md=prf" % word]
    for url in urls:
        result = json.loads(urllib.urlopen(url).read())
        words = [w["word"] for w in result if pos_datamuse in w["tags"]]
    
    
    
    if not words:
        words = [word]
    
    def format_word(w):
        w = justlowerletters(w)
        if pos[:1]=="V":
            w = conjugate_verb(w, pos)
        if pos[:1]=="N":
            w = conjugate_noun(w, pos)
        return w

    
    if word_additions:
        # add extra words. (ex. profanity)
        words = word_additions[:]
    
    # mix it up
    shuffle(words)
        
    # conjugate, lowercase, etc
    words = map(format_word, words)
    return words

algo("big red dog", 
         alliteration_only = True, 
         number_of_examples = 100, 
         force_phoneme = [], 
         flatten = False, 
         split_by = "sentences", # "sentences" or "punctuation"
         normalize = False, 
         try_to_avoid_dups = True, 
         word_additions = [],
         swap_probability = 1.0,
         hyper_profanity = False,
         ignore_numbers = True,
         debug = True)

