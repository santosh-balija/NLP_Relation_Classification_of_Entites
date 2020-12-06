#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn


nlp = spacy.load('en_core_web_sm')
wnl = WordNetLemmatizer()
train_file = open("train.txt", "r")  
test_file = open("test.txt", "r")
all_pos_tags_between_bigrams = {}
all_ner_tags_entities = {}
all_before_after = {}
levin_class = defaultdict(set)
all_hypernyms_of_entites = {}
all_root_verbs = {}
all_entities = {}
all_pos_tags_of_entites = {}
all_words_between = {}


# In[2]:


################################## Function for Pre processing ###################################################################

def preProcessTheLines(all_lines, train_sentences, train_e_1, train_e_2, train_relations, words_between_entites):
    for i,line in enumerate(all_lines):
        if i % 2 == 0:
            sentence_number_removed = line.split('\t')[1]
            line_strip = sentence_number_removed.strip()
            processed_line = line_strip[1:len(line_strip)-1].split(' ')
            final_sentence = ''
            k = 0
            found_e_1 = False
            found_e_2 = False
            words_between = []
            while k < len(processed_line):
                if processed_line[k] == '<e1>':
                    found_e_1 = True
                    e_1 = ''
                    j = k+1
                    while processed_line[j] != '</e1>':
                        e_1 = e_1 + ' ' + processed_line[j]
                        j = j + 1
                    train_e_1.append([e_1.strip(),k-1])
                    final_sentence = final_sentence + ' ' + e_1.strip()
                    k = j+1
                elif processed_line[k] == '<e2>':
                    found_e_2 = True
                    e_2 = ''
                    j = k+1
                    while processed_line[j] != '</e2>':
                        e_2 = e_2 + ' ' + processed_line[j]
                        j = j + 1
                    train_e_2.append([e_2.strip(),k-3])
                    final_sentence = final_sentence + ' ' + e_2.strip()
                    k = j+1
                elif processed_line[k] != '</e1>' and processed_line[k] != '</e2>':
                    if found_e_1 == True and found_e_2 == False:
                        words_between.append([processed_line[k],k-3])
                    final_sentence = final_sentence + ' ' + processed_line[k]
                    k = k + 1
            train_sentences.append(final_sentence.strip())
            words_between_entites.append(words_between)
        else:
            train_relations.append(line) 
    return all_lines, train_sentences, train_e_1, train_e_2, train_relations, words_between_entites


# In[3]:


####################################### Function to Tokenize ####################################################################

# To tokenize the sentences 
# input  ==> list of sentences to tokenize
# output ==> list of tokenized sentences

def tokenize_the_sentences(train_sentences):
    tokenized_sentences = []
    for sentences in train_sentences:
        tokens = nltk.word_tokenize(sentences.strip())
        tokenized_sentences.append(tokens)
    return tokenized_sentences


# In[4]:


############################################### Function to Lemmatize ###########################################################

# To lemmatize the tokens generated
# input  ==> list of tokenized sentences
# output ==> list of lemmatized words

def lemmatize_the_sentences(tokenized_sentences):
    wnl = WordNetLemmatizer()
    lemmatized_words = []

    for tokens in tokenized_sentences:
        lemmatized_word = []
        for token in tokens:
            lemma_word = wnl.lemmatize(token)
            lemmatized_word.append(lemma_word)
        lemmatized_words.append(lemmatized_word)
    return lemmatized_words


# In[5]:


########################################## Function for Pos Tags ################################################################

# To generate the Pos tags for the lemmatized words
# input  ==> list of lemmatized words
# output ==> list of Pos tags of each of the lemmatized words

def generate_pos_tags(lemmatized_words):
    pos_tags = []
    for lemmatized_word in lemmatized_words:
        pos_tags.append(nltk.pos_tag(lemmatized_word))
    return pos_tags


# In[6]:


################################### Function for of the Pos Tags between Entites ################################################

# To generate the Pos Tags of the words in between the entities
# input  ==> list of words in between the entites, pos tags of the lemmatized words
# output ==> pos tags of the words in between the entites

def generate_pos_tags_in_between_the_entites( words_between_entites, pos_tags ):
    pos_tags_between = []
    for i,words_between in enumerate(words_between_entites):
        if len(words_between_entites[i]) > 0:
            start_index = words_between_entites[i][0][1]
            end_index = words_between_entites[i][-1][1]
            pos_tags_between.append(pos_tags[i][start_index:end_index+1])
        else:
            pos_tags_between.append(())
    return pos_tags_between


# In[7]:


######################################### Function for Dependency ###############################################################

def generate_dependeny(train_sentences):
    dependancy = []
    for sentence in train_sentences:
        sentence_dependency = []
        doc = nlp(sentence)
        for item in doc:
            sentence_dependency.append([item.text, item.dep_, item.head.text, item.head.pos_])
        dependancy.append(sentence_dependency)
    return dependancy


# In[8]:


########################################## Function for Ner Tags ##############################################################

# To generate the ner tags of the sentences
# input  ==> list of sentences to obtain the ner tags
# output ==> list of ner tags of the sentences

def generate_ner_tags(train_sentences):
    ner_list = []
    for sentence in train_sentences:
        ner_sentence = {}
        doc = nlp(sentence)
        for entity in doc.ents:
            ner_sentence[entity.text] = entity.label_
        ner_list.append(ner_sentence)
    return ner_list


# In[9]:


################################# Function for length of words Between entites ##################################################
# feature 1
# input  ==> list of words between the entites in a sentence
# output ==> length or number of words between those entities

def calculate_words_in_between(words_between_entites):
    length_of_words_in_between = []
    for words_between in words_between_entites:
        length_of_words_in_between.append(len(words_between))
    return length_of_words_in_between


# In[10]:


######################################## Function for pos Tags bigrams ##########################################################

# input  ==> list pos tags of the words between the entites
# output ==> one hot pos tag list of words between the entites 

def generate_pos_tags_bigrams_between_entites( pos_tags_between ):
    pos_strings = []
    for i,tags_between in enumerate(pos_tags_between):
        pos_string = ''
        for k in range(len(tags_between)):
            pos_string = pos_string + ' ' +tags_between[k][1][:2]
        pos_strings.append(pos_string.strip())
    pos_tags_between_bigrams = []
    for pos_string in pos_strings:
        pos_string_list = ['<s>']
        pos_string_split = pos_string.split(' ')
        pos_string_list = pos_string_list + pos_string_split + ['</s>']
        d = {}
        for i in range(len(pos_string_list)-1):
            d[(pos_string_list[i], pos_string_list[i+1])] = d.get((pos_string_list[i], pos_string_list[i+1]), 0 ) + 1
        pos_tags_between_bigrams.append(d)
    return pos_tags_between_bigrams


# In[ ]:





# In[11]:


############################################ One Hot of Pos Tags of Between entites #############################################

# feature 2
# To obtain the one hot labels for the pos_tags_between_bigrams
# input  ==> pos tags between the entites 
# output ==> one hot encoding of the pos tags between the entites

def generate_one_hot_pos_tags_between(pos_tags_between_bigrams):

    one_hot_pos_tags_between = []

    for pos_tag_between_bigram in pos_tags_between_bigrams:
        one_hot = [0]*(len(all_pos_tags_between_bigrams)+1)
        for bigram in pos_tag_between_bigram:
            if bigram in all_pos_tags_between_bigrams:
                one_hot[all_pos_tags_between_bigrams[bigram]] = 1
            else:
                one_hot[-1] = 1
        one_hot_pos_tags_between.append(one_hot)
    return one_hot_pos_tags_between


# In[ ]:





# In[12]:


################################# Function for Ner Tags #########################################################################

def generate_ner_tags_of_entities(train_e_1, ner_list, train_e_2):
    ner_tags_of_enties = []
    for i in range(len(train_e_1)):
        ner_tags_of_entity = ['NULL']*2
        for entity_key in ner_list[i]:
            if train_e_1[i][0] in entity_key or entity_key in train_e_1[i][0]:
                ner_tags_of_entity[0] = ner_list[i][entity_key]
            if train_e_2[i][0] in entity_key or entity_key in train_e_2[i][0]:
                ner_tags_of_entity[1] = ner_list[i][entity_key]
        ner_tags_of_enties.append(ner_tags_of_entity)
    return ner_tags_of_enties


# In[13]:


############################################# One Hot fpr NER Tags ##############################################################

def generate_one_hot_ners(ner_tags_of_entites):
    one_hot_ner_1 = []
    one_hot_ner_2 = []
    for pair in ner_tags_of_entites:
        pair_one_hot_ner_1 = [0]*(len(all_ner_tags_entities)+1)
        pair_one_hot_ner_2 = [0]*(len(all_ner_tags_entities)+1)
        
        if pair[0] in all_ner_tags_entities:
            pair_one_hot_ner_1[all_ner_tags_entities[pair[0]]] = 1
        else:
            pair_one_hot_ner_1[-1] = 1
        
        if pair[1] in pair_one_hot_ner_2:
            pair_one_hot_ner_2[all_ner_tags_entities[pair[1]]] = 1
        else:
            pair_one_hot_ner_2[-1] = 1
        
        one_hot_ner_1.append(pair_one_hot_ner_1)
        one_hot_ner_2.append(pair_one_hot_ner_2)
    return (one_hot_ner_1, one_hot_ner_2)    


# In[14]:


############################################## Function for Before After ########################################################

# To generate the words before and after entites one and two
# input  ==> sentences to extract the before and after words
# output ==> list of 'before after' string in lemma form
def generate_before_after(train_sentences):
    
    before_after = []
    wnl = WordNetLemmatizer()
    for i,sentence in enumerate(train_sentences):
        end_of_before = train_sentences[i].find(train_e_1[i][0])
        start_to_before = sentence[:end_of_before].strip()
        before_word = wnl.lemmatize(start_to_before.split(' ')[-1])
        start_of_after = train_sentences[i].find(train_e_2[i][0]) + len(train_e_2[i][0])
        after_to_end = sentence[start_of_after:].strip()
        after_word =  wnl.lemmatize(after_to_end.split(' ')[0])
        before_after.append(before_word+'_'+after_word)
        
    return before_after


# In[15]:


################################################### Function for pos Tags of Entites ############################################

def get_pos_tags_of_entites(pos_tags, train_e_1):
    pos_tags_e_1 = []
    for i,pos_tag in enumerate(pos_tags):
        pos_tags_e_1.append(pos_tag[train_e_1[i][1]][1])
    return pos_tags_e_1


# In[16]:


######################################## Function for One Hot of Before After ###################################################
# To generate the one hot labels of the before after words
# input  ==> list of before after words
# output ==> one hot labels of the before ofter words

def generate_one_hot_encoded_before_after(before_after):
    one_hot_encoded_before_after = []
    for word in before_after:
        one_hot_word = [0]*(len(all_before_after)+1)
        if word in all_before_after:
            one_hot_word[all_before_after[word]] = 1
        else:
            one_hot_word[-1] = 1
        one_hot_encoded_before_after.append(one_hot_word)

    return one_hot_encoded_before_after


# In[17]:


#################################################### Function for shortest Dependency Path ######################################

# To generate the shortedt dependency of the list of sentences
# input  ==> list of sentences
# output ==> list of the short dependency p of the sentences given

import networkx as nx

def generate_shortest_dependency_path(train_sentences):
    short_dependency_path = []

    for i,sentence in enumerate(train_sentences):
        doc = nlp(sentence)
        edges = []
        for token in doc:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        entity_1 = ''
        entity_2 = ''
        if train_e_1[i][0].lower() in set(graph.nodes):
            entity_1 = train_e_1[i][0].lower()
        else:
            entity_split = train_e_1[i][0].split()
            for entity in entity_split:
                if entity.lower() in set(graph.nodes):
                    entity_1 = entity.lower()
                    break

        if train_e_2[i][0].lower() in set(graph.nodes):
            entity_2 = train_e_2[i][0].lower()
        else:
            entity_split = train_e_2[i][0].split()
            for entity in entity_split:
                if entity.lower() in set(graph.nodes):
                    entity_2 = entity.lower()
                    break
        if entity_1 != '' and entity_2 != '':
            try:
                short_dependency_path.append(nx.shortest_path(graph, source=entity_1, target=entity_2))
            except nx.NetworkXNoPath:
                short_dependency_path.append([])
        else:
            short_dependency_path.append([])
    return short_dependency_path


# In[18]:


############################################## Function to get the Root Verbs ###################################################

# To get the root verbs of all the sentences given
# input  ==> list of sentences to obtain the root verbs
# output ==> list of root verbs of the sentences given

def get_root_verbs(train_sentences):
    root_verbs_of_all_sentences = []

    for sentence in train_sentences:
        doc = nlp(sentence)

        # check if the token and its head are same( for head )
        for token in doc:
            if token.text == token.head.text:
                root_verbs_of_all_sentences.append(token.text)
                break
    return root_verbs_of_all_sentences 


# In[19]:


######################################## Function to get the root Verbs of the sentences ####################################### 

def get_one_hot_root_verbs(root_verbs_of_all_sentences):
    
    one_hot_root_verbs = []
    for root_verb in root_verbs_of_all_sentences:
        one_hot = [0]*len(all_root_verbs)
        if wnl.lemmatize(root_verb) in all_root_verbs:
            one_hot[all_root_verbs[wnl.lemmatize(root_verb)]] = 1
        one_hot_root_verbs.append(one_hot)
        
    return one_hot_root_verbs


# In[20]:


############################################ Function to get the One Hot levinClasses of sentence Root Verbs ################### 

def get_one_hot_levin_classes(root_verbs_of_all_sentences, short_dependency_path):
    one_hot_levin_classes = []
    for i,root_verb in enumerate(root_verbs_of_all_sentences):

        # initialize the levin class one hot vector for each sentence
        # last index is used for dummy class in a levin classification
        one_hot_levin_class = [0]*(len(levin_class)+1)


        # check if the lamda form of the verb is in the short dependency path
        # else give it a dummy class of the levin classification
        if root_verb in set(short_dependency_path[i]):
            root = nlp(root_verb)
            lambda_form_of_verb = ''

            for word in root:
                lambda_form_of_verb = word.lemma_

            for key in levin_class:
                if lambda_form_of_verb in levin_class[key]:
                    one_hot_levin_class[key[1]] = 1
                else:
                    one_hot_levin_class[-1] = 1
        else:
            one_hot_levin_class[-1] = 1

        one_hot_levin_classes.append(one_hot_levin_class)
    return one_hot_levin_classes


# In[21]:


def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


# In[22]:


############################################# Function to get the hypernyms opf e1 and e2 #######################################

def generate_hypernyms_of_entites(pos_tags, train_e_1, train_e_2):
    
    hypernyms = []
    for pos_tag_sentence in pos_tags:
        hypernym_of_sentence = {}
        for pos_tag_word in pos_tag_sentence:
            hypernyms_list_word = []
            for ss in wn.synsets(pos_tag_word[0],pos=penn2morphy(pos_tag_word[1])):
                hypernyms_list_word = hypernyms_list_word + ss.hypernyms()

            hypernym_of_sentence[pos_tag_word] = hypernyms_list_word[:1]
        hypernyms.append(hypernym_of_sentence)
        
    hypernyms_of_e_1 = []
    for i in range(len(train_e_1)):
        hypernyms_of_sentence = []
        for entity_key in hypernyms[i]:
            if entity_key[0] in train_e_1[i][0]:
                hypernyms_of_sentence.append(hypernyms[i][entity_key])
        hypernyms_of_e_1.append(hypernyms_of_sentence)

    final_hypernyms_of_e_1 = []
    for hypernym_sentence in hypernyms_of_e_1:
        temp = []
        for hypernym_list in hypernym_sentence:
            temp = temp + hypernym_list
        final_hypernyms_of_e_1.append(temp[-1:len(temp)])
    
    hypernyms_of_e_2 = []
    for i in range(len(train_e_2)):
        hypernyms_of_sentence = []
        for entity_key in hypernyms[i]:
            if entity_key[0] in train_e_2[i][0]:
                hypernyms_of_sentence.append(hypernyms[i][entity_key])
        hypernyms_of_e_2.append(hypernyms_of_sentence)
        
    final_hypernyms_of_e_2 = []
    for hypernym_sentence in hypernyms_of_e_2:
        temp = []
        for hypernym_list in hypernym_sentence:
            temp = temp+hypernym_list
        final_hypernyms_of_e_2.append(temp[-1:len(temp)])
    
    return final_hypernyms_of_e_1, final_hypernyms_of_e_2


# In[23]:


################################ Function to get the One Hot of Hypernyms #######################################################

def one_hot_of_hypernyms(final_hypernyms):
    one_hot_hypernym_encoded = []
    for hypernym_list in final_hypernyms:
        temp = [0]*(len(all_hypernyms_of_entites)+1)
        if len(hypernym_list) > 0:
            if hypernym_list[0] in all_hypernyms_of_entites:
                temp[all_hypernyms_of_entites[hypernym_list[0]]] = 1
            else:
                temp[-1] = 1
        one_hot_hypernym_encoded.append(temp)
    return one_hot_hypernym_encoded


# In[24]:


########################################### Function for One Hot of the root Verb ###############################################

def one_hot_position_of_root_verb(dependancy, train_sentences, train_e_1, train_e_2):
    
    index_root_verb = []
    for sentence_dependancy in dependancy:
        for i, word_dependancy in enumerate(sentence_dependancy):
            if(word_dependancy[1]=='ROOT'):
                index_root_verb.append(i)
                
    index_e_1_list = []
    index_e_2_list = []
    
    for i,sentence in enumerate(train_sentences): 
        if train_e_1[i][0] == '' and train_e_2[i][0] == '':
            index_e_1 = 0
            index_e_2 = len(sentence)-1
            
        elif train_e_2[i][0] == '':
            index_e_2 = sentence.split(' ').index(train_e_1[i][0].split()[0])
        elif train_e_1[i][0] == '':
            index_e_1 = sentence.split(' ').index(train_e_2[i][0].split()[0])
        else:
            index_e_1 = sentence.split(' ').index(train_e_1[i][0].split()[0])
            index_e_2 = sentence.split(' ').index(train_e_2[i][0].split()[0])
        index_e_1_list.append(index_e_1)
        index_e_2_list.append(index_e_2)
        
    position_of_root_verb = []
    for i in range(len(train_sentences)):
        if(index_root_verb[i] < index_e_1_list[i]):
            position_of_root_verb.append([1,0,0])
        elif index_root_verb[i] > index_e_2_list[i]:
            position_of_root_verb.append([0,0,1])
        else:
            position_of_root_verb.append([0,1,0])
    return position_of_root_verb


# In[25]:


def get_one_hot_of_entities(entities):
    one_hot_entites = []
    for entity in entities:
        one_hot = [0]*(len(all_entities)+1)
        if entity[0] in all_entities:
            one_hot[all_entities[entity[0]]] = 1
        else:
            one_hot[-1] = 1
        one_hot_entites.append(one_hot)
    return one_hot_entites


# In[26]:


def get_one_hot_pos_tag_entity(pos_tags_e_1):
    one_hot_tags = []
    for tag in pos_tags_e_1:
        one_hot = [0]*(len(all_pos_tags_of_entites)+1)
        if tag in all_pos_tags_of_entites:
            one_hot[all_pos_tags_of_entites[tag]] = 1
        else:
            one_hot[-1] = 1
        one_hot_tags.append(one_hot)
    return one_hot_tags


# In[27]:


def generate_one_hot_words_between(words_between):
    one_hot_words_between = []
    for words in words_between:
        one_hot = [0]*(len(all_words_between))
        for word in words:
            if word[0] in all_words_between:
                one_hot[all_words_between[word[0]]] = 1
          
        one_hot_words_between.append(one_hot)
    return one_hot_words_between


# In[28]:


# train Pre Processing

train_sentences = []
train_e_1 = []
train_e_2 = []
train_relations = []
all_lines = []
words_between_entites = []
for line in train_file:
    if line.rstrip():
        all_lines.append(line.strip())

all_lines, train_sentences, train_e_1, train_e_2, train_relations, words_between_entites = preProcessTheLines(all_lines, train_sentences, train_e_1, train_e_2, train_relations, words_between_entites)  


# In[29]:


# test Pre Processing

test_sentences = []
test_e_1 = []
test_e_2 = []
test_relations = []
all_test_lines = []
words_between_entites_test = []
for line in test_file:
    if line.rstrip():
        all_test_lines.append(line.strip())

all_test_lines, test_sentences, test_e_1, test_e_2, test_relations, words_between_entites_test = preProcessTheLines(all_test_lines, test_sentences, test_e_1, test_e_2, test_relations, words_between_entites_test) 


# In[30]:


index = 0
for words in words_between_entites:
    for w in words:
        if w[0] not in all_words_between:
            all_words_between[w[0]] = index
            index = index + 1


# In[31]:


one_hot_words_between = generate_one_hot_words_between(words_between_entites)

one_hot_words_between_test = generate_one_hot_words_between(words_between_entites_test)


# In[32]:


# len(one_hot_words_between[0])


# In[33]:


index = 0
for i in range(len(train_sentences)):
    if train_e_1[i][0] not in all_entities:
        all_entities[train_e_1[i][0]] = index
        index = index + 1
    if train_e_2[i][0] not in all_entities:
        all_entities[train_e_2[i][0]] = index
        index = index + 1


# In[34]:


# index = 0
# for i in range(len(test_sentences)):
#     if test_e_1[i] not in all_test_lines:
#         all_test_lines[test_e_1[i]] = index
#         index = index + 1
#     if test_e_2[i] not in all_test_lines:
#         all_entities[test_e_2[i]] = index
#         index = index + 1


# In[35]:


#################################################### Tokenization ###############################################################

# for train data
tokenized_sentences = tokenize_the_sentences(train_sentences)

# for test data
tokenized_sentences_test = tokenize_the_sentences(test_sentences)


# In[36]:


################################################ lemmatized_words ###############################################################
# train data

lemmatized_words = lemmatize_the_sentences(tokenized_sentences)
# test data
lemmatized_words_test = lemmatize_the_sentences(tokenized_sentences_test)


# In[37]:


############################################### Pos Tags ########################################################################

# pos tags of the train data
pos_tags = generate_pos_tags(lemmatized_words)

# pos_tags of the test data
pos_tags_test = generate_pos_tags(lemmatized_words_test)


# In[ ]:





# In[38]:


############################################### Get the Pos Tags of Entites #####################################################

pos_tags_e_1 = get_pos_tags_of_entites(pos_tags, train_e_1)
pos_tags_e_2 = get_pos_tags_of_entites(pos_tags, train_e_2)


all_entity_tags = set(pos_tags_e_1)|set(pos_tags_e_2)

index = 0
for tag in all_entity_tags:
    if tag not in all_pos_tags_of_entites:
        all_pos_tags_of_entites[tag] = index
        index = index + 1

pos_tags_e_1_test = get_pos_tags_of_entites(pos_tags_test, test_e_1)
pos_tags_e_2_test = get_pos_tags_of_entites(pos_tags_test, test_e_2)


# In[39]:


################################################### Ner Tags ####################################################################

# train data
ner_list = generate_ner_tags(train_sentences)

# test data
ner_list_test = generate_ner_tags(test_sentences)


# In[40]:


############################################### Dependency ######################################################################

# for train Data
dependency = generate_dependeny(train_sentences)

# for test Data
dependency_test = generate_dependeny(test_sentences)


# In[41]:


########################################  Pos Tags between the Entites ##########################################################

# train data
pos_tags_between = generate_pos_tags_in_between_the_entites( words_between_entites, pos_tags )

# for test data
pos_tags_between_test = generate_pos_tags_in_between_the_entites( words_between_entites_test, pos_tags_test )


# In[42]:


########################################### Bigrams of Pos Tags in Between Entites ##############################################

# train Data
pos_tags_between_bigrams = generate_pos_tags_bigrams_between_entites(pos_tags_between)


############### perform One Hot Encoding and store the indexes in a dictionary "all_pos_tags_between_bigrams" ###################
index = 0
for pos_tag_between_bigram in pos_tags_between_bigrams:
    for bigram in pos_tag_between_bigram:
        if bigram not in all_pos_tags_between_bigrams:
            all_pos_tags_between_bigrams[bigram] = index
            index = index + 1

# test Data
pos_tags_between_bigrams_test = generate_pos_tags_bigrams_between_entites(pos_tags_between_test)


# In[43]:


###############################################  Feature 1 ######################################################################

# Generate the number of words in between the entites for train and test data

# for train data
length_of_words_in_between = calculate_words_in_between(words_between_entites)

# for test data
length_of_words_in_between_test = calculate_words_in_between(words_between_entites_test)


# In[44]:


length_of_words_in_between[0:10]


# In[45]:


#############################################  Feature 2 ########################################################################

# Generate the one Hot of the pos Tags in between the entites ( of length 443 for train and test data )

# one Hot of pos Tags in between the Entites 
one_hot_pos_tags_between = generate_one_hot_pos_tags_between(pos_tags_between_bigrams)

# one Hot of pos Tags in between the Entites 
one_hot_pos_tags_between_test = generate_one_hot_pos_tags_between(pos_tags_between_bigrams)


# In[46]:


############################################### Ner Tags of the Entites #########################################################


# train data
ner_tags_of_entites = generate_ner_tags_of_entities(train_e_1, ner_list, train_e_2)
# all_entites_ner = []
index = 0
for ner_pair in ner_tags_of_entites:
    if ner_pair[0] not in all_ner_tags_entities:
        all_ner_tags_entities[ner_pair[0]] = index
        index = index + 1
    if ner_pair[1] not in all_ner_tags_entities:
        all_ner_tags_entities[ner_pair[1]] = index
        index = index + 1

        
# test 
ner_tags_of_entites_test = generate_ner_tags_of_entities(test_e_1, ner_list_test, test_e_2)


# In[ ]:





# In[47]:


############################################# Feature 3 #########################################################################

# train data
one_hot_ner_1, one_hot_ner_2  = generate_one_hot_ners(ner_tags_of_entites)

# test data
one_hot_ner_1_test, one_hot_ner_2_test  = generate_one_hot_ners(ner_tags_of_entites_test)


# In[48]:


# print(len(one_hot_ner_2_test), len(one_hot_ner_2))


# In[49]:


############################################# Before After words of Entites #####################################################

# train Data
before_after = generate_before_after(train_sentences)
index = 0
for word in before_after:
    if word not in all_before_after:
        all_before_after[word] = index
        index = index + 1

# test Data    
before_after_test = generate_before_after(test_sentences)      
# print(len(before_after))


# In[50]:


# print(len(before_after_test))


# In[51]:


########################################## Feature 4 ############################################################################


# train Data
one_hot_encoded_before_after = generate_one_hot_encoded_before_after(before_after)

# test Data
one_hot_encoded_before_after_test = generate_one_hot_encoded_before_after(before_after_test)
# print(len(one_hot_encoded_before_after[0]))


# In[52]:


# print(len(one_hot_encoded_before_after_test[0]))


# In[53]:


########################################## Short Dependency Path ################################################################

# train Data
short_dependency_path = generate_shortest_dependency_path(train_sentences)
# test Data
short_dependency_path_test = generate_shortest_dependency_path(test_sentences)


# In[54]:


# print(len(short_dependency_path_test))


# In[55]:


########################################### Dictionary To Store LevinClasses ####################################################

levin_class_file = open("LevinClass.txt", "r")  
class_name = ''
class_number = -1
for line in levin_class_file:
    if line.rstrip():
        stripped_line = line.strip()
        split_line = stripped_line.split(' ')
        first_char = split_line[0][0]
        if first_char.isdigit():
            class_name = split_line[0]
            class_number = class_number + 1
        else:
            for verb in split_line:
                levin_class[(class_name, class_number)].add(verb)


# In[56]:


########################################## Get the Root Verbs ###################################################################

# train Data
root_verbs_of_all_sentences = get_root_verbs(train_sentences)


# perform one Hot of the root verb using a dictionary "all_root_verbs"
index = 0
for root_verb in root_verbs_of_all_sentences:
    if wnl.lemmatize(root_verb) not in all_root_verbs:
        all_root_verbs[wnl.lemmatize(root_verb)] = index
        index = index + 1
        
# test Data
root_verbs_of_all_sentences_test = get_root_verbs(test_sentences) 


# In[57]:


# print(len(all_root_verbs))


# In[58]:


################################################ Feature 5 ######################################################################

# for train Data
one_hot_levin_classes = get_one_hot_levin_classes(root_verbs_of_all_sentences, short_dependency_path)

# for test Data
one_hot_levin_classes_test = get_one_hot_levin_classes(root_verbs_of_all_sentences_test, short_dependency_path_test)
# print(len(one_hot_levin_classes[0]))


# In[59]:


########################################### Hypernyms of Entities ###############################################################

# train Data
final_hypernyms_of_e_1, final_hypernyms_of_e_2 = generate_hypernyms_of_entites(pos_tags, train_e_1, train_e_2)


# perform one Hot of hypernyms using a dictionary "all_hypernyms_of_entites"

flattened_set_e_1 = set(x for l in final_hypernyms_of_e_1 for x in l)
flattened_set_e_2 = set(x for l in final_hypernyms_of_e_2 for x in l)
total_flattened_set = flattened_set_e_1|flattened_set_e_2
index = 0
for synset in total_flattened_set:
    all_hypernyms_of_entites[synset] = index
    index = index + 1
    
# test Data
final_hypernyms_of_e_1_test, final_hypernyms_of_e_2_test = generate_hypernyms_of_entites(pos_tags_test, test_e_1, test_e_2)


# In[60]:


# print(len(final_hypernyms_of_e_1_test))
# print(len(all_words_between_bigrams))


# In[61]:


############################################ Feature 6 ##########################################################################

# for train Data
one_hot_hypernym_encoded_e_1 = one_hot_of_hypernyms(final_hypernyms_of_e_1)

# for test Data
one_hot_hypernym_encoded_e_1_test = one_hot_of_hypernyms(final_hypernyms_of_e_1_test)


# In[62]:


# print(len(one_hot_hypernym_encoded_e_1_test))


# In[63]:


################################################### Feature 7 ###################################################################

# for train Data
one_hot_hypernym_encoded_e_2 = one_hot_of_hypernyms(final_hypernyms_of_e_2)

# for test Data
one_hot_hypernym_encoded_e_2_test = one_hot_of_hypernyms(final_hypernyms_of_e_2_test)


# In[64]:


################################################## Feature 8 ####################################################################

# for train Data
one_hot_root_verb = get_one_hot_root_verbs(root_verbs_of_all_sentences)

# for test Data
one_hot_root_verb_test = get_one_hot_root_verbs(root_verbs_of_all_sentences_test)


# In[65]:


################################################ Feature 9 ######################################################################

# for train Data
one_hot_root_verb_position = one_hot_position_of_root_verb(dependency, train_sentences, train_e_1, train_e_2)

# for test Data
one_hot_root_verb_position_test = one_hot_position_of_root_verb(dependency_test, test_sentences, test_e_1, test_e_2)


# In[66]:


# print(len(one_hot_root_verb_position_test))


# In[67]:


################################################## Feature 10 ###################################################################

# for train data
one_hot_of_e_1 = get_one_hot_of_entities(train_e_1)
one_hot_of_e_2 = get_one_hot_of_entities(train_e_2)

# for test data
one_hot_of_e_1_test = get_one_hot_of_entities(test_e_1)
one_hot_of_e_2_test = get_one_hot_of_entities(test_e_2)


# In[68]:


############################################## Feature 11 #######################################################################


# for train Data
one_hot_pos_tag_e_1 = get_one_hot_pos_tag_entity(pos_tags_e_1)
one_hot_pos_tag_e_2 = get_one_hot_pos_tag_entity(pos_tags_e_2)


# for test Data
one_hot_pos_tag_e_1_test = get_one_hot_pos_tag_entity(pos_tags_e_1_test)
one_hot_pos_tag_e_2_test = get_one_hot_pos_tag_entity(pos_tags_e_2_test)


# In[ ]:





# In[69]:


# #############################################  Feature 12 #######################################################################

# # Generate the one Hot of the pos Tags in between the entites ( of length 443 for train and test data )

# # one Hot of pos Tags in between the Entites 
# one_hot_pos_tags_between = generate_one_hot_words_between(pos_tags_between_bigrams)

# # one Hot of pos Tags in between the Entites 
# one_hot_pos_tags_between_test = generate_one_hot_words_between(pos_tags_between_bigrams)


# In[70]:


################################################### Features for lable ##########################################################


train_features = []

for i in range(len(train_sentences)):
    feature = []
    feature.append([length_of_words_in_between[i]])
    feature.append(one_hot_pos_tags_between[i])
    feature.append(one_hot_ner_1[i])
    feature.append(one_hot_ner_2[i])
    feature.append(one_hot_encoded_before_after[i])
    feature.append(one_hot_levin_classes[i])
    feature.append(one_hot_hypernym_encoded_e_1[i])
    feature.append(one_hot_hypernym_encoded_e_2[i])
    feature.append(one_hot_root_verb_position[i])
    feature.append(one_hot_root_verb[i])
    feature.append(one_hot_of_e_1[i])
    feature.append(one_hot_of_e_2[i])
    feature.append(one_hot_pos_tag_e_1[i])
    feature.append(one_hot_pos_tag_e_2[i])
    feature.append(one_hot_words_between[i])
    
    train_features.append(feature)

test_features = []

for i in range(len(test_sentences)):
    feature = []
    feature.append([length_of_words_in_between_test[i]])
    feature.append(one_hot_pos_tags_between_test[i])
    feature.append(one_hot_ner_1_test[i])
    feature.append(one_hot_ner_2_test[i])
    feature.append(one_hot_encoded_before_after_test[i])
    feature.append(one_hot_levin_classes_test[i])
    feature.append(one_hot_hypernym_encoded_e_1_test[i])
    feature.append(one_hot_hypernym_encoded_e_2_test[i])
    feature.append(one_hot_root_verb_position_test[i])
    feature.append(one_hot_root_verb_test[i])
    feature.append(one_hot_of_e_1_test[i])
    feature.append(one_hot_of_e_2_test[i])
    feature.append(one_hot_pos_tag_e_1[i])
    feature.append(one_hot_pos_tag_e_2[i])
    feature.append(one_hot_words_between_test[i])
    
    test_features.append(feature)


# In[71]:


def flatten_the_features(features):
    flattend_features = []
    for i,one_sample in enumerate(features):
        one_sample_features = []
        for feature in one_sample:
            for f in feature:
                one_sample_features.append(f)
        flattend_features.append(one_sample_features)
    return flattend_features

flattened_train_features = flatten_the_features(train_features)    
flattened_test_features = flatten_the_features(test_features)
# print(len(flattened_train_features[0]))


# In[72]:


print(len(flattened_test_features[0]))


# In[73]:


################################################ Extract Label and Direction ####################################################


def extract_label_and_direction(train_relations):
    only_relation = []
    only_direction = []
    for relation in train_relations:
        strip_relation = relation.strip()
        split_relation = relation.split('(')
        only_relation.append(split_relation[0])
        if len(split_relation) < 2:
            only_direction.append(2)
        elif split_relation[1][:-1] == 'e1,e2':
            only_direction.append(0)
        elif split_relation[1][:-1] == 'e2,e1':
            only_direction.append(1)
    return only_relation, only_direction


# for train Data
only_relation_train, only_direction_train = extract_label_and_direction(train_relations)

# To assign a unique value to each class in the train Data 
relations_dict = {}
class_number = 1
for relation in only_relation_train:
    if relation not in relations_dict:
        relations_dict[relation] = class_number 
        class_number = class_number + 1 
        
# for test Data
only_relation_test, only_direction_test = extract_label_and_direction(test_relations)
        

relations_dict_combined = {}
class_number = 1
for relation in train_relations:
    if relation not in relations_dict_combined:
        relations_dict_combined[relation] = class_number 
        class_number = class_number + 1 


# In[74]:


# print(len(only_relation_train))


# In[75]:


##################################################### Labels ####################################################################

# for train Data

# for seperate models
train_labels = []
for relation in only_relation_train:
    train_labels.append(relations_dict[relation])

# for combined model
train_labels_combined = []
for relation in train_relations:
    train_labels_combined.append(relations_dict_combined[relation])
    
    
# for test Data
test_labels = []
for relation in only_relation_test:
    test_labels.append(relations_dict[relation])

# for combined models
test_labels_combined = []
for relation in test_relations:
    test_labels_combined.append(relations_dict_combined[relation])

# for seperate models
train_direction = only_direction_train
test_direction = only_direction_test


# In[76]:


# print(len(relations_dict))


# In[77]:


############################################### Features for Direction ##########################################################

train_features_for_direction = []

for i in range(len(train_sentences)):
    feature = []
    feature.append([length_of_words_in_between[i]])
    feature.append(one_hot_levin_classes[i])
    feature.append(one_hot_root_verb_position[i])
    feature.append(one_hot_root_verb[i])
    train_features_for_direction.append(feature)

test_features_for_direction = []

for i in range(len(test_sentences)):
    feature = []
    feature.append([length_of_words_in_between_test[i]])
    feature.append(one_hot_levin_classes_test[i])
    feature.append(one_hot_root_verb_position_test[i])
    feature.append(one_hot_root_verb_test[i])
    test_features_for_direction.append(feature)


# In[78]:


flattened_train_features_for_direction = flatten_the_features(train_features_for_direction)    
flattened_test_features_for_direction  = flatten_the_features(test_features_for_direction)


# In[79]:


######################################## csr matrix for relation ################################################################


row_relation = []
col_relation = []
data_relation = []
from tqdm.notebook import tqdm
for i in tqdm(range(len(flattened_train_features))):
    for j in range(len(flattened_train_features[0])):
        if flattened_train_features[i][j] == 1:
            row_relation.append(i)
            col_relation.append(j)
            data_relation.append(1)
# print(len(row_relation), len(col_relation), len(data_relation))


# In[80]:


from scipy.sparse import csr_matrix
train_x = csr_matrix( (data_relation, (row_relation, col_relation)), shape=(len(flattened_train_features), len(flattened_train_features[0])) )
from sklearn.model_selection import GridSearchCV
from sklearn import svm

clf_relation = svm.SVC(decision_function_shape='ovo', kernel='linear', gamma = 0.01, C = 1)
clf_relation.fit(train_x, train_labels)


# from sklearn import svm
# from scipy.sparse import csr_matrix
# from sklearn.model_selection import GridSearchCV



# train_x = csr_matrix( (data_relation, (row_relation, col_relation)), shape=(len(flattened_train_features), len(flattened_train_features[0])) )

# params_grid = {'C': [0.1, 1, 10, 100], 'kernel':['linear','poly','rbf'] }


# grid_clf1 = GridSearchCV(svm.SVC(gamma='auto'), params_grid, cv=5, return_train_score= False)

# grid_clf1.fit(train_x, train_labels)


# In[85]:


######################################## csr matrix for direction ###############################################################

row_direction = []
col_direction = []
data_direction = []

for i in tqdm(range(len(flattened_train_features_for_direction))):
    for j in range(len(flattened_train_features_for_direction[0])):
        if flattened_train_features_for_direction[i][j] == 1:
            row_direction.append(i)
            col_direction.append(j)
            data_direction.append(1)
print(len(row_direction), len(col_direction), len(data_direction))


# In[86]:


# from sklearn import svm

# clf_2 = svm.SVC(decision_function_shape='ovo', kernel='poly', gamma=0.1, C=1)
# clf_2.fit(flattened_train_features_for_direction, only_direction)


from scipy.sparse import csr_matrix
train_x_direction = csr_matrix((data_direction, (row_direction, col_direction)), shape=(len(flattened_train_features_for_direction), len(flattened_train_features_for_direction[0])) )
from sklearn.model_selection import GridSearchCV
from sklearn import svm

clf_direction = svm.SVC(decision_function_shape='ovo', kernel='rbf', gamma = 0.1, C = 1)
clf_direction.fit(train_x_direction, train_direction)


# from sklearn import svm
# from scipy.sparse import csr_matrix
# from sklearn.model_selection import GridSearchCV


# train_x_direction = csr_matrix((data_direction, (row_direction, col_direction)), shape=(len(flattened_train_features_for_direction), len(flattened_train_features_for_direction[0])) )

# params_grid = {'C': [0.1, 1, 10, 100], 'kernel':['linear','poly','rbf'] }


# grid_clf2 = GridSearchCV(svm.SVC(gamma='auto'), params_grid, cv=5, return_train_score= False)

# grid_clf2.fit(train_x_direction, train_direction)


# In[91]:


# combined model for direction and relation

from scipy.sparse import csr_matrix
train_x = csr_matrix( (data_relation, (row_relation, col_relation)), shape=(len(flattened_train_features), len(flattened_train_features[0])) )
from sklearn.model_selection import GridSearchCV
from sklearn import svm

clf_combined = svm.SVC(decision_function_shape='ovo', kernel='linear', gamma = 0.01, C = 1)
clf_combined.fit(train_x, train_labels_combined)


# In[81]:


import pickle
filename = 'NLPmodel_relation.sav'
pickle.dump(clf_relation, open(filename, 'wb'))


# In[87]:


import pickle
filename = 'NLPmodel_direction.sav'
pickle.dump(clf_direction, open(filename, 'wb'))


# In[92]:


import pickle
filename = 'NLPmodel_combined.sav'
pickle.dump(clf_combined, open(filename, 'wb'))


# In[106]:


##################################################### Task 2 ####################################################################

# Tokenise

print(tokenized_sentences[0:10])


# In[107]:


# Lemmatise

print(lemmatized_words[0:10])


# In[108]:


# Pos Tagging

print(pos_tags[0:10])


# In[111]:


# Dependency 

print(dependency[0:1])


# In[114]:


# Hypernyms
print(" Hypernyms od e1")
print(final_hypernyms_of_e_1[0:10])

print("Hypernyms of e2")
print(final_hypernyms_of_e_2[0:10])


# In[115]:


# Ner Tags

print(ner_list[0:10])


# In[82]:


################################################# Predicted Relation ############################################################
import time
start_time = time.time()
predicted_relation = []
for i in range(len(flattened_test_features)):
    p = clf_relation.predict([flattened_test_features[i]])
    predicted_relation.append(p[0])
current_time = time.time()
elapsed_time_relation = current_time - start_time


# In[ ]:





# In[88]:


################################################## Predicted Direction ##########################################################
import time
start_time = time.time()
predicted_direction = []
for i in range(len(flattened_test_features_for_direction)):
    p = clf_direction.predict([flattened_test_features_for_direction[i]])
    predicted_direction.append(p[0])
current_time = time.time()
elapsed_time_direction = current_time - start_time


# In[93]:


############################################ Prediction for combined model ######################################################

import time
start_time = time.time()
predicted = []
for i in range(len(flattened_test_features)):
    p = clf_combined.predict([flattened_test_features[i]])
    predicted.append(p[0])
current_time = time.time()
elapsed_time = current_time - start_time
print(elapsed_time)


# In[84]:


############################################### Correctly classified Relations###################################################

classified_correct_relation = 0
for i in range(len(test_labels)):
    if test_labels[i] == predicted_relation[i]:
        classified_correct_relation = classified_correct_relation + 1


# In[90]:


############################################## Correctly Classified Direction ###################################################

classified_correct_direction = 0
for i in range(len(test_labels)):
    if test_direction[i] == predicted_direction[i]:
        classified_correct_direction = classified_correct_direction + 1


# In[94]:


############################################## Both correctly classified #####################################################

both_classified_correctly = 0
for i in range(len(test_labels_combined)):
    if test_labels_combined[i] == predicted[i]:
        both_classified_correctly = both_classified_correctly + 1


# In[121]:





# In[ ]:


################################################## Combined Model ###############################################################

#'NLPmodel_combined.sav'

################################################ Model only for relation #######################################################

#

############################################### Model only for direction ########################################################

#


# In[83]:


############################################### Only for relation ###############################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("model for relation")
print("------------------------------------------------------")
print("Accuracy ")
print(accuracy_score(predicted_relation, test_labels))
print("------------------------------------------------------")
print("Precision Score")
print(precision_score(predicted_relation, test_labels, average='macro'))
print("------------------------------------------------------")
print("Recall Score")
print(recall_score(predicted_relation, test_labels, average='macro'))
print("------------------------------------------------------")
print(" F1 Score ")
print(f1_score(predicted_relation, test_labels, average='macro'))
print("------------------------------------------------------")
print("Time Elapsed")
print(elapsed_time_relation)


# In[89]:


##################################################### Only for direction ########################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("model for direction")
print("------------------------------------------------------")
print("Accuracy ")
print(accuracy_score(predicted_direction, test_direction))
print("------------------------------------------------------")
print("Precision Score")
print(precision_score(predicted_direction, test_direction, average='macro'))
print("------------------------------------------------------")
print("Recall Score")
print(recall_score(predicted_direction, test_direction, average='macro'))
print("------------------------------------------------------")
print(" F1 Score ")
print(f1_score(predicted_direction, test_direction, average='macro'))
print("------------------------------------------------------")
print("Time Elapsed")
print(elapsed_time_direction)


# In[95]:


############################################# Combined model ###################################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Accuracy ")
print(accuracy_score(predicted, test_labels_combined))
print("------------------------------------------------------")
print("Precision Score")
print(precision_score(predicted, test_labels_combined, average='macro'))
print("------------------------------------------------------")
print("Recall Score")
print(recall_score(predicted, test_labels_combined, average='macro'))
print("------------------------------------------------------")
print(" F1 Score ")
print(f1_score(predicted, test_labels_combined, average='macro'))
print("------------------------------------------------------")
print("Time Elapsed")
print(elapsed_time)


# In[96]:


# test Pre Processing

new_sentences = []
new_e_1 = []
new_e_2 = []
new_relations = []
all_lines = ['2	" The 2008 Ohio Bobcats football team represented <e1> Ohio University </e1> during the 2008 <e2> NCAA </e2> Division I FBS football season . "']
new_words_between = []

all_lines, new_sentences, new_e_1, new_e_2, new_relations, new_words_between = preProcessTheLines(all_lines, new_sentences, new_e_1, new_e_2, new_relations, new_words_between) 


# In[97]:


length_of_words_in_between_new = calculate_words_in_between(new_sentences)


# In[98]:


new_tokenized_sentences = tokenize_the_sentences(new_sentences)
new_lemmatized_words = lemmatize_the_sentences(new_tokenized_sentences)
new_pos_tags = generate_pos_tags(new_lemmatized_words)
new_pos_tags_between = generate_pos_tags_in_between_the_entites( new_words_between, new_pos_tags )
new_pos_tags_between_bigrams = generate_pos_tags_bigrams_between_entites(new_pos_tags_between)
new_one_hot_pos_tags_between = generate_one_hot_pos_tags_between(new_pos_tags_between_bigrams)


# In[99]:


new_ner_list = generate_ner_tags(new_sentences)
new_ner_tags_of_entites = generate_ner_tags_of_entities(new_e_1, new_ner_list, new_e_2)
new_one_hot_ner_1, new_one_hot_ner_2  = generate_one_hot_ners(new_ner_tags_of_entites)


# In[100]:


new_before_after = generate_before_after(new_sentences)
new_one_hot_encoded_before_after = generate_one_hot_encoded_before_after(new_before_after)


# In[101]:


new_root_verbs_of_all_sentences = get_root_verbs(new_sentences)
new_short_dependency_path = generate_shortest_dependency_path(new_sentences)
new_one_hot_levin_classes = get_one_hot_levin_classes(new_root_verbs_of_all_sentences, new_short_dependency_path)


# In[102]:


new_final_hypernyms_of_e_1, new_final_hypernyms_of_e_2 = generate_hypernyms_of_entites(new_pos_tags, new_e_1, new_e_2)
new_one_hot_hypernym_encoded_e_1 = one_hot_of_hypernyms(new_final_hypernyms_of_e_1)
# one_hot_hypernym_encoded_e_1_test = one_hot_of_hypernyms(final_hypernyms_of_e_1_test)
new_one_hot_hypernym_encoded_e_2 = one_hot_of_hypernyms(new_final_hypernyms_of_e_2)


# In[103]:


new_dependency = generate_dependeny(new_sentences)
new_one_hot_root_verb_position = one_hot_position_of_root_verb(new_dependency, new_sentences, new_e_1, new_e_2)


# In[104]:


new_one_hot_root_verb = get_one_hot_root_verbs(new_root_verbs_of_all_sentences)


# In[105]:


new_one_hot_of_e_1 = get_one_hot_of_entities(new_e_1)
new_one_hot_of_e_2 = get_one_hot_of_entities(new_e_2)


# In[106]:


new_pos_tags_e_1 = get_pos_tags_of_entites(new_pos_tags, new_e_1)
new_pos_tags_e_2 = get_pos_tags_of_entites(new_pos_tags, new_e_2)

new_one_hot_pos_tag_e_1 = get_one_hot_pos_tag_entity(new_pos_tags_e_1)
new_one_hot_pos_tag_e_2 = get_one_hot_pos_tag_entity(new_pos_tags_e_2)


# In[107]:


new_one_hot_words_between = generate_one_hot_words_between(new_words_between)


# In[108]:


new_feature = []
new_feature_direction = []

for i in range(len(new_sentences)):
    feature = []
    direction_feature = []
    feature.append([length_of_words_in_between_new[i]])
    feature.append(new_one_hot_pos_tags_between[i])
    feature.append(new_one_hot_ner_1[i])
    feature.append(new_one_hot_ner_2[i])
    feature.append(new_one_hot_encoded_before_after[i])
    feature.append(new_one_hot_levin_classes[i])
    feature.append(new_one_hot_hypernym_encoded_e_1[i])
    feature.append(new_one_hot_hypernym_encoded_e_2[i])
    feature.append(new_one_hot_root_verb_position[i])
    feature.append(new_one_hot_root_verb[i])
    feature.append(new_one_hot_of_e_1[i])
    feature.append(new_one_hot_of_e_2[i])
    feature.append(new_one_hot_pos_tag_e_1[i])
    feature.append(new_one_hot_pos_tag_e_2[i])
    feature.append(new_one_hot_words_between[i])
    
    
    direction_feature.append([length_of_words_in_between_new[i]])
    direction_feature.append(new_one_hot_levin_classes[i])
    direction_feature.append(new_one_hot_root_verb_position[i])
    direction_feature.append(new_one_hot_root_verb[i])
    
    new_feature.append(feature)
    new_feature_direction.append(direction_feature)


# In[118]:


new_flat_features = flatten_the_features(new_feature)
new_flat_direction_features = flatten_the_features(new_feature_direction)


# In[110]:


len(new_flat_features[0])


# In[115]:


# using combined svm model for relation and direction

print("Using combined SVM model ")
print("------------------------------------------------")
combined_model = pickle.load(open('NLPmodel_combined.sav', 'rb'))
result = combined_model.predict([new_flat_features[0]])

for key in relations_dict_combined:
    if relations_dict_combined[key] == result[0]:
        print("Classified as ", key)


# In[116]:


# using svm model  only for relation

print(" Using SVM model only for relation")
print("------------------------------------------------")
relation_model = pickle.load(open('NLPmodel_relation.sav', 'rb'))
result = relation_model.predict([new_flat_features[0]])

for key in relations_dict:
    if relations_dict[key] == result[0]:
        print(" Classified relation is ", key)


# In[120]:


# using svm model  only for direction

print(" Using SVM model only for direction")
print("--------------------------------------------------")
direction_model = pickle.load(open('NLPmodel_direction.sav', 'rb'))
result = direction_model.predict([new_flat_direction_features[0]])

# print(" Classified as ")
if result[0] == 0:
    print(" Classified as (e1,e2)")
elif result[0] == 1:
    print(" Classified as (e1,e2)")
else:
    print(" Classified as 'No direction'")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




