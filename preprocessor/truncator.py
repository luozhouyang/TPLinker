import abc

from transformers import BertTokenizerFast


class AbstractExampleTruncator(abc.ABC):
    """Truncate examples whose text is too long. *THIS MAY BREAK RELATIONS!*"""

    @abc.abstractmethod
    def truncate(self, example, **kwargs):
        raise NotImplementedError()

    def _adjust_entity_list(self, example, start, end, token_offset, char_offset, **kwargs):
        entity_list = []
        for entity in example['entity_list']:
            # print(entity)
            token_span = entity.get('tok_span', None) or entity.get('token_span', None)
            char_span = entity['char_span']
            if not token_span:
                continue
            if start > token_span[0] or end < token_span[1]:
                continue
            entity_list.append({
                'text': entity['text'],
                'type': entity['type'],
                'tok_span': [token_span[0] - token_offset, token_span[1] - token_offset],
                'char_span': [char_span[0] - char_offset, char_span[1] - char_offset]
            })
        return entity_list

    def _adjust_relation_list(self, example, start, end, token_offset, char_offset, **kwargs):
        relation_list = []
        for relation in example['relation_list']:
            subj_token_span, obj_token_span = relation['subj_tok_span'], relation['obj_tok_span']
            subj_char_span, obj_char_span = relation['subj_char_span'], relation['obj_char_span']
            if start <= subj_token_span[0] and subj_token_span[1] <= end and start <= obj_token_span[0] and obj_token_span[1] <= end:
                relation_list.append({
                    'subj_tok_span': [subj_token_span[0] - token_offset, subj_token_span[1] - token_offset],
                    'obj_tok_span': [obj_token_span[0] - token_offset, obj_token_span[1] - token_offset],
                    'subj_char_span': [subj_char_span[0] - char_offset, subj_char_span[1] - char_offset],
                    'obj_char_span': [obj_char_span[0] - char_offset, obj_char_span[1] - char_offset],
                    'subject': relation['subject'],
                    'object': relation['object'],
                    'predicate': relation['predicate'],
                })
        return relation_list


class BertExampleTruncator(AbstractExampleTruncator):

    def __init__(self, pretrained_bert_path, max_sequence_length=100, **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_bert_path, add_special_tokens=False, do_lower_case=False)

    def truncate(self, example, **kwargs):
        all_examples = []

        text = example['text']
        codes = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        if len(codes['input_ids']) < self.max_sequence_length:
            return [example]

        tokens = self.tokenizer.convert_ids_to_tokens(codes['input_ids'])
        offset = codes['offset_mapping']

        for start in range(0, len(tokens), self.max_sequence_length//2):
            # do not truncte word pieces
            while str(tokens[start]).startswith('##'):
                start -= 1
            end = start + self.max_sequence_length
            char_spans = offset[start: end]
            char_span = [char_spans[0][0], char_spans[-1][1]]
            text_subs = text[char_span[0]:char_span[1]]

            token_offset = start
            char_offset = char_span[0]

            all_examples.append({
                'text': text_subs,
                'entity_list': self._adjust_entity_list(example, start, end, token_offset, char_offset),
                'relation_list': self._adjust_relation_list(example, start, end, token_offset, char_offset)
            })

        return all_examples
