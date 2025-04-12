import json

import jieba
import spacy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(base_url='https://chatnio.cdreader.vip/v1', max_retries=1024, timeout=1600)
llm = 'qwen-plus'
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
    "云琛",
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
    "临州",
    "时微"
]


def split_words(text: str, stopwords: list) -> list:
    known_names.extend(extract_names(text))
    known_names.extend(llm_based_extract_names(text))
    stopwords.extend(known_names)
    for w in stopwords:
        text = text.replace(w, '_')
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list if len(w) > 1 and '_' not in w and not w.isnumeric()]


def extract_names(text: str) -> list:
    names = [x.text for x in ner_model(text).ents if x.label_ in ['PERSON']]
    print('SpaCy NER:', names)
    return names


def llm_based_extract_names(text: str) -> list:
    role = '你是一个写作经验丰富的人物角色设计师'
    output_restraint = '你的输出只能在同一行, JSONL格式, 请确保 JSONL 的格式合法.'
    system_prompt = {
        '你是': role,
        '输出数据格式': 'JSONL',
        '输出格式细节': ['角色名1', '角色名2', '角色名3', '角色名4', '...'],
        '输出格式限制': output_restraint
    }
    prompt = {
        '任务目标': '请你仔细阅读[文段], 找出文段中所提及的所有命名实体(角色), 包括主角, 配角.',
        '文段': text
    }
    response = openai_client.chat.completions.create(
        model=llm,
        messages=[
            {'role': 'system', 'content': json.dumps(system_prompt, ensure_ascii=False)},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.1,
        top_p=0.2
    ).choices[0].message.content
    names = json.loads(response[response.find('['): response.find(']') + 1])
    print('GPT NER:', names)
    return names
