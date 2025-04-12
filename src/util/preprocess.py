import jieba
import spacy

ner_model = spacy.load("zh_core_web_sm")

known_names = [
    "楚辞",
    "季烟",
    "津宸",
    "厉寒",
    "乐珊",
    "星晚",
    "凌远",
    "鹿溪",
    "明微",
    "庭琛",
    "厉庭琛",
    "北安",
    "凉薇",
    "雨清",
    "季成德",
    "澄映",
    "念安",
    "青柠",
    "易辰",
    "余清舒",
    "舒念",
    "南音",
    "延深",
    "鹤年",
    "临州"
]


def split_words(text: str, stopwords: list) -> list:
    stopwords.extend(extract_names(text))
    for w in stopwords:
        text = text.replace(w, '_')
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = [w for w in seg_list if w not in stopwords and w not in known_names and len(w) > 1]
    seg_list = [w for w in seg_list if '_' not in w]
    seg_list = [w for w in seg_list if not w.isnumeric()]
    return seg_list


def extract_names(text: str) -> list:
    names = [x.text for x in ner_model(text).ents if x.label_ in ['PERSON']]
    print(names)
    return names
