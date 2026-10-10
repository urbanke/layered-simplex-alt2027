"""Offline checks for strict source pins, explicit recipes, and plain inputs."""
import importlib.util
import json
from pathlib import Path

import pytest

from lsa.alt.bible import load_corpus


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/get_kjv.py'
spec = importlib.util.spec_from_file_location('prepare_corpus', SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def sample():
    raw = (prep.START + '\n' + prep.HEADING +
           '\n1:1 Adam’s sons’ named A-B -- C. ***\n' + prep.END).encode()
    primary = (prep.HEADING + ' Adam ’s sons named A B C .').encode()
    secondary = 'Adam ’ s sons ’ named A - B — C . * * *\n'.encode()
    records = []
    for name, value in zip(('kjv', 'kjv-secondary'), (primary, secondary), strict=True):
        records.append({'dataset_id': name, 'file': f'data/{name}.txt',
                        'sha256': prep.digest(value), 'source_sha256': [prep.digest(raw)],
                        'bytes': len(value), 'tokens': len(value.split()),
                        'distinct_types': len(set(value.decode().split()))})
    return raw, records, primary, secondary


def test_exact_recipes_and_plain_loader(tmp_path):
    raw, records, primary, secondary = sample()
    result = prep.prepare(raw, records, tmp_path)
    assert [Path(row['path']).read_bytes() for row in result] == [primary, secondary]
    assert prep.prepare(raw, records, tmp_path) == result
    (tmp_path / 'manifest.json').write_text(json.dumps(records[0]))
    tokens, _ = load_corpus({'corpus': 'kjv.txt', 'manifest': 'manifest.json'}, tmp_path)
    assert tokens == primary.decode().split()
    (tmp_path / 'kjv.txt').write_bytes(primary + b' changed')
    with pytest.raises(ValueError, match='checksum'):
        load_corpus({'corpus': 'kjv.txt', 'manifest': 'manifest.json'}, tmp_path)


def test_changed_download_or_prepared_hash_writes_nothing(tmp_path):
    raw, records, _, _ = sample()
    with pytest.raises(ValueError, match='downloaded source checksum'):
        prep.prepare(raw + b' changed', records, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()
    records[1]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='prepared checksum'):
        prep.prepare(raw, records, tmp_path / 'bad-tokenization')
    assert not (tmp_path / 'bad-tokenization').exists()


def test_existing_different_output_is_never_overwritten(tmp_path):
    raw, records, _, _ = sample()
    (tmp_path / 'kjv-secondary.txt').write_bytes(b'preserve me')
    with pytest.raises(FileExistsError):
        prep.prepare(raw, records, tmp_path)
    assert (tmp_path / 'kjv-secondary.txt').read_bytes() == b'preserve me'
    assert not (tmp_path / 'kjv.txt').exists()


def test_secondary_requires_declared_formatting_features():
    raw, _, _, _ = sample()
    with pytest.raises(ValueError, match='double hyphen'):
        prep.secondary(raw.decode().replace('--', '—'))
    with pytest.raises(ValueError, match='three-star'):
        prep.secondary(raw.decode().replace('***\n' + prep.END, '\n' + prep.END))
