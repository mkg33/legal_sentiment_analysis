def test_chunk_paragraphs_with_token_counts_preserves_token_total():
    from transformers import AutoTokenizer
    from legal_emotion.corpus import (
        chunk_paragraphs_with_token_counts,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-bert',
        use_fast=True,
    )
    paragraph = 'hello ' * 200
    max_length = 32
    stride = 8
    chunks = chunk_paragraphs_with_token_counts(
        [paragraph],
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    assert chunks
    special = tokenizer.num_special_tokens_to_add(pair=False)
    max_body = max(
        1,
        max_length - special,
    )
    assert all((c.body_tokens <= max_body for c in chunks))
    assert all((c.new_body_tokens <= c.body_tokens for c in chunks))
    assert (
        chunks[0].new_body_tokens == chunks[0].body_tokens
    )
    total_new = sum((c.new_body_tokens for c in chunks))
    full_len = len(tokenizer(
            paragraph,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )['input_ids'])
    assert total_new == full_len


def test_chunk_paragraphs_with_token_counts_stride_zero_equals_body():
    from transformers import AutoTokenizer
    from legal_emotion.corpus import (
        chunk_paragraphs_with_token_counts,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-bert',
        use_fast=True,
    )
    paragraph = 'hello ' * 200
    max_length = 32
    chunks = chunk_paragraphs_with_token_counts(
        [paragraph],
        tokenizer,
        max_length=max_length,
        stride=0,
    )
    assert chunks
    assert all((c.new_body_tokens == c.body_tokens for c in chunks))
