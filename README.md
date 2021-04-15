# TPLinker

论文 [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://www.aclweb.org/anthology/2020.coling-main.138.pdf) 的PyTorch实现。

## 所需依赖

* pytorch
* pytorch-lightning


## 训练数据

整理好的NYT数据集下载：[NYT](https://huaichen-oss.oss-cn-hangzhou.aliyuncs.com/public/tplinker-bert-nyt.zip?versionId=CAEQDxiBgMCKm83SxhciIDFmNmY1OGZiMzc0YzRhMDY4ODBmZTEyNDhlOTJmYTg3)

> 下载之后解压，放到当前项目的 `data/` 目录下。

也可以用下述命令下载：
```bash
mkdir data && cd data
wget -O tplinker-bert-nyt.zip https://huaichen-oss.oss-cn-hangzhou.aliyuncs.com/public/tplinker-bert-nyt.zip?versionId=CAEQDxiBgMCKm83SxhciIDFmNmY1OGZiMzc0YzRhMDY4ODBmZTEyNDhlOTJmYTg3

unzip tplinker-bert-nyt.zip

```


训练数据格式如下：

```bash
{"text": "In Queens , North Shore Towers , near the Nassau border , supplanted a golf course , and housing replaced a gravel quarry in Douglaston .", "id": "valid_0", "relation_list": [{"subject": "Douglaston", "object": "Queens", "subj_char_span": [125, 135], "obj_char_span": [3, 9], "predicate": "/location/neighborhood/neighborhood_of", "subj_tok_span": [26, 28], "obj_tok_span": [1, 2]}, {"subject": "Queens", "object": "Douglaston", "subj_char_span": [3, 9], "obj_char_span": [125, 135], "predicate": "/location/location/contains", "subj_tok_span": [1, 2], "obj_tok_span": [26, 28]}], "entity_list": [{"text": "Douglaston", "type": "DEFAULT", "char_span": [125, 135], "tok_span": [26, 28]}, {"text": "Queens", "type": "DEFAULT", "char_span": [3, 9], "tok_span": [1, 2]}, {"text": "Queens", "type": "DEFAULT", "char_span": [3, 9], "tok_span": [1, 2]}, {"text": "Douglaston", "type": "DEFAULT", "char_span": [125, 135], "tok_span": [26, 28]}]}
```

## 训练模型

> 相关的参数在 `tplinker/run_tplinker.py` 文件直接修改即可。

```bash
nohup python -m tplinker.run_tplinker --gpus=0 >> train.log 2>&1 &
```
