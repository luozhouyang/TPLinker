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

    def _adjust_offset_mapping(self, offset_mapping, char_offset, max_sequence_length=100, **kwargs):
        offsets = []
        for start, end in offset_mapping:
            offsets.append([start - char_offset, end - char_offset])
        # padding to max_sequence_length to avoid DataLoader runtime error
        while len(offsets) < max_sequence_length:
            offsets.append([0, 0])
        return offsets


class BertExampleTruncator(AbstractExampleTruncator):

    def __init__(self, tokenizer: BertTokenizerFast, max_sequence_length=100, window_size=50, **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.window_size = window_size
        self.tokenizer = tokenizer

    def truncate(self, example, **kwargs):
        all_examples = []

        text = example['text']
        codes = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        if len(codes['input_ids']) < self.max_sequence_length:
            all_examples.append({
                'text': text,
                'entity_list': example['entity_list'],
                'relation_list': example['relation_list'],
                'offset_mapping': self._adjust_offset_mapping(codes['offset_mapping'], 0),
                'token_offset': 0,
                'char_offset': 0,
            })
            return all_examples

        tokens = self.tokenizer.convert_ids_to_tokens(codes['input_ids'])
        offset = codes['offset_mapping']

        for start in range(0, len(tokens), self.window_size):
            # do not truncte word pieces
            while str(tokens[start]).startswith('##'):
                start -= 1
            end = min(start + self.max_sequence_length, len(tokens))
            range_offset_mapping = offset[start: end]
            char_span = [range_offset_mapping[0][0], range_offset_mapping[-1][1]]
            text_subs = text[char_span[0]:char_span[1]]

            token_offset = start
            char_offset = char_span[0]

            truncated_example = {
                'text': text_subs,
                'entity_list': self._adjust_entity_list(example, start, end, token_offset, char_offset),
                'relation_list': self._adjust_relation_list(example, start, end, token_offset, char_offset),
                'token_offset': token_offset,
                'char_offset': char_offset,
                'offset_mapping': self._adjust_offset_mapping(range_offset_mapping, char_offset)
            }
            all_examples.append(truncated_example)

            if end > len(tokens):
                break

        return all_examples
