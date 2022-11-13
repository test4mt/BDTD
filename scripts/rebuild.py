import spacy
import json

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
print('Loading clusters...')

with open("../asset/sense_inventory/auto_clusters.json", 'r', encoding='utf8') as auto:
    auto_clusters = json.load(auto)
with open("../asset/sense_inventory/manual_clusters.json", 'r', encoding='utf8') as manual:
    manual_clusters = json.load(manual)


def extract_attribute_sets(entry, poly_word):
    attribute_sets = dict()
    for gloss in entry["[SOURCE GLOSSES]"]:
        line_nlp = nlp(gloss)
        for tok in line_nlp:
            cur_text = tok.text.lower()
            lexeme = nlp.vocab[cur_text]
            if cur_text == poly_word:
                continue
            if not lexeme.is_stop and not lexeme.is_punct:
                if not attribute_sets.get(cur_text, None):
                    attribute_sets[cur_text] = 1
                else:
                    attribute_sets[cur_text] += 1
    for synonym in entry["[SYNONYMS]"]:
        for synonym_part in synonym.split('_'):
            synonym_part = synonym_part.lower()
            lexeme = nlp.vocab[synonym_part]
            if synonym_part == poly_word:
                continue
            if not lexeme.is_stop:
                if not attribute_sets.get(synonym_part, None):
                    attribute_sets[synonym_part] = 1
                else:
                    attribute_sets[synonym_part] += 1
    sense_attributes = sorted(attribute_sets.items(),
                              key=lambda item: item[1], reverse=True)[0:10]
    if len(sense_attributes) < 10:
        print(sense_attributes)
    return sense_attributes


# rebuild manual clusters by complementing gloss, synonym and attribute words
for term in auto_clusters.keys():
    for cs_id1 in auto_clusters[term].keys():
        for sense in auto_clusters[term][cs_id1]["[SENSES]"]:
            for cs_id2 in manual_clusters[term].keys():
                if sense in manual_clusters[term][cs_id2]["[SENSES]"]:
                    if not manual_clusters[term][cs_id2].get("[SOURCE GLOSSES]", None):
                        manual_clusters[term][cs_id2]["[SOURCE GLOSSES]"] = auto_clusters[term][cs_id1][
                            "[SOURCE GLOSSES]"]
                    else:
                        manual_clusters[term][cs_id2]["[SOURCE GLOSSES]"].extend(
                            auto_clusters[term][cs_id1]["[SOURCE GLOSSES]"])
                    if not manual_clusters[term][cs_id2].get("[SYNONYMS]", None):
                        manual_clusters[term][cs_id2]["[SYNONYMS]"] = auto_clusters[term][cs_id1]["[SYNONYMS]"]
                    else:
                        manual_clusters[term][cs_id2]["[SYNONYMS]"].extend(auto_clusters[term][cs_id1]["[SYNONYMS]"])
                    if not manual_clusters[term][cs_id2].get("[TARGET GLOSSES]", None):
                        manual_clusters[term][cs_id2]["[TARGET GLOSSES]"] = auto_clusters[term][cs_id1][
                            "[TARGET GLOSSES]"]
                    else:
                        manual_clusters[term][cs_id2]["[TARGET GLOSSES]"].extend(
                            auto_clusters[term][cs_id1]["[TARGET GLOSSES]"])

for term in manual_clusters.keys():
    for cs_id in manual_clusters[term].keys():
        manual_clusters[term][cs_id]["[SENSES]"] = list(set(manual_clusters[term][cs_id]["[SENSES]"]))
        manual_clusters[term][cs_id]["[SOURCE GLOSSES]"] = list(set(manual_clusters[term][cs_id]["[SOURCE GLOSSES]"]))
        manual_clusters[term][cs_id]["[SYNONYMS]"] = list(set(manual_clusters[term][cs_id]["[SYNONYMS]"]))
        manual_clusters[term][cs_id]["[TARGET GLOSSES]"] = list(set(manual_clusters[term][cs_id]["[TARGET GLOSSES]"]))
        manual_clusters[term][cs_id]["[ATTRIBUTE SETS]"] = extract_attribute_sets(manual_clusters[term][cs_id], term)
        if len(manual_clusters[term][cs_id]["[ATTRIBUTE SETS]"]) < 10:
            print(manual_clusters[term][cs_id]["[SENSES]"])

with open("../asset/sense_inventory/sense_dict.json", 'w', encoding='utf8') as out_fo:
    json.dump(manual_clusters, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

print('Done')