"""Read-only comparison of the author's key, WAV files and MATLAB metadata.

Run with the project's Python. Writes only outputs/author_stimulus_audit.
No fuzzy matches or assumed equivalence between different audio files.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import wave
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

from bat_internship.bat_data import BatData
from bat_internship.paths import MAT_PATH

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'Autorendaten'
OUT = ROOT / 'outputs' / 'author_stimulus_audit'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def key_rows():
    """Read cell values and retain Excel row provenance without editing Excel."""
    with zipfile.ZipFile(AUTHOR / 'KeyforbatsStims.xlsx') as z:
        strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        result = []
        for sheet in (1, 2):
            rows = []
            for row in ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml')).findall('.//s:row', NS):
                cells = {}
                for c in row.findall('s:c', NS):
                    v = c.find('s:v', NS)
                    if v is not None:
                        cells[''.join(filter(str.isalpha, c.attrib['r']))] = strings[int(v.text)] if c.attrib.get('t') == 's' else v.text
                rows.append((int(row.attrib['r']), cells))
            result.append(rows)
        return result


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio_roots = [p.parent for p in AUTHOR.rglob('Vocalization') if p.is_dir()]
    if len(audio_roots) != 1:
        raise ValueError(f'Expected one audio root: {audio_roots}')
    audio_root = audio_roots[0]
    wavs, by_path = [], {}
    for p in sorted(AUTHOR.rglob('*.wav')):
        with wave.open(str(p), 'rb') as w:
            frames = w.readframes(w.getnframes())
            row = dict(path=p.relative_to(ROOT).as_posix(), folder=p.parent.relative_to(audio_root).as_posix(),
                       sample_rate_hz=w.getframerate(), channels=w.getnchannels(), sample_width_bytes=w.getsampwidth(),
                       frames=w.getnframes(), duration_s=w.getnframes()/w.getframerate(),
                       pcm_sha256=hashlib.sha256(frames).hexdigest(), file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        wavs.append(row)
        by_path[row['path']] = row
    vocal, morph = key_rows()
    specs = [('vocal12', 'B', 12, 'Vocalization', vocal), ('vocal17', 'C', 17, 'Vocalization', vocal),
             ('extended22', 'D', 22, 'VocalizationExtended', vocal), ('reverse22', 'E', 22, 'Reverse', vocal),
             ('morph40', 'B', 40, 'Morphing', morph)]
    mappings = []
    for key, col, count, folder, rows in specs:
        indices = []
        for excel_row, cells in rows:
            if excel_row == 1 or col not in cells:
                continue
            idx = int(cells[col])
            indices.append(idx)
            name = cells['A']
            candidate, status = name, 'exact'
            if not candidate.lower().endswith('.wav'):
                candidate += '.wav'
                status = 'extension_added'
            path = (audio_root / folder / candidate).relative_to(ROOT).as_posix()
            if path not in by_path and key == 'extended22' and name.startswith('voc trial '):
                path = (audio_root / folder / (name + '_norm.wav')).relative_to(ROOT).as_posix()
                status = 'normalized_variant_needs_confirmation'
            if path not in by_path:
                path, status = '', 'missing'
            audio = by_path.get(path, {})
            mappings.append(dict(stimulus_set=key, index_matlab=idx, index_python=idx-1,
                excel_sheet='Morphing' if key == 'morph40' else 'VocalizationandReverse', excel_row=excel_row,
                key_filename=name, key_category=cells.get('F', cells.get('D', '')),
                morph_sequence=cells.get('C', '') if key == 'morph40' else '',
                wav_path=path, match_status=status, pcm_sha256=audio.get('pcm_sha256', ''),
                sample_rate_hz=audio.get('sample_rate_hz', ''), duration_s=audio.get('duration_s', '')))
        if sorted(indices) != list(range(1,count+1)):
            raise ValueError(f'Non-contiguous or duplicate indices for {key}: {indices}')
    mappings.sort(key=lambda r:(r['stimulus_set'],r['index_matlab']))
    sets = {key: sorted([r for r in mappings if r['stimulus_set']==key], key=lambda r:r['index_matlab']) for key,*_ in specs}
    families = [('VocalizationMat','VocalizationCategory','VocalizationMatDimension'),
                ('VocalizationMatExtended','VocalizationExtendedCategory','VocalizationMatExtendedDimension'),
                ('VocalizationMatReverse','VocalizationReverseCategory','VocalizationMatReverseDimension'),
                ('MorphingMat','MorphingMatCategory','MorphingMatDimension')]
    sites, assigned, errors = [], [], []
    with BatData(MAT_PATH) as data:
        for i in range(data.n_sites):
            bat, site = int(data.field(i,'BatId').item()), int(data.field(i,'SiteNb').item())
            for field, catfield, dimfield in families:
                d = data.dataset(i, field)
                empty = bool(d.attrs.get('MATLAB_empty',0))
                scalar_nan = bool(not empty and d.size == 1 and d.dtype.kind == 'f' and np.isnan(d[()]).all())
                row = dict(site_index_python=i, site_number=site, bat_id=bat, matrix=field,
                           empty=empty, scalar_nan=scalar_nan, shape_h5py=str(tuple(d.shape)), stimulus_set='', category_match='')
                if not empty and not scalar_nan:
                    row['category_match']=True
                    categories = list(data.field(i,catfield).ravel())
                    labels = list(data.field(i,dimfield).ravel())
                    expected_labels = ['ROI number', 'Morphing Vocalization stimulus' if field=='MorphingMat' else 'Vocalization stimulus', 'Repetition number', 'Frame number']
                    if labels != expected_labels or d.ndim != 4 or len(categories)!=d.shape[2] or d.shape[3]!=data.field(i,'RoiNb').item() or d.shape[0]!=data.field(i,'FrametoSecond').size:
                        raise ValueError(f'Metadata/axis mismatch {i} {field}')
                    key = {'VocalizationMat':f'vocal{d.shape[2]}','VocalizationMatExtended':'extended22',
                           'VocalizationMatReverse':'reverse22','MorphingMat':'morph40'}[field]
                    if len(sets[key])!=d.shape[2]: raise ValueError('Stimulus count mismatch')
                    row['stimulus_set']=key
                    for m, actual in zip(sets[key], categories):
                        if key=='morph40':
                            seq=m['morph_sequence'].split()[-1]
                            desc=m['key_category']
                            if desc.startswith('Morph '): expected=f'Morphing continuum {seq} {desc.split()[1]}'
                            else: expected=desc.replace(' exemplar','')+' '+seq
                        else:
                            expected={'social':'Social','navigation':'Navigation','USV':'Mouse USVs','white noise squence':'White Noise temporal match'}[m['key_category'].replace(' reverse','')]
                        ok=actual==expected
                        row['category_match'] &= ok
                        if not ok: errors.append(dict(site=i,matrix=field,index=m['index_matlab'],expected=expected,actual=actual))
                        assigned.append(dict(site_index_python=i,site_number=site,bat_id=bat,matrix=field,
                                             **m,matlab_category=actual,category_match=ok))
                sites.append(row)
    used={m['wav_path'] for m in mappings if m['wav_path']}
    groups=defaultdict(list)
    for w in wavs:
        w['referenced_by_key']=w['path'] in used
        groups[(w['sample_rate_hz'],w['channels'],w['sample_width_bytes'],w['pcm_sha256'])].append(w['path'])
    summary=dict(wav_count=len(wavs), key_rows=len(mappings), assigned_rows=len(assigned),
                 referenced_wav_count=len(used), unreferenced_wav_count=len(wavs)-len(used),
                 match_counts=dict(Counter(m['match_status'] for m in mappings)),
                 category_errors=errors, matrix_counts=dict(Counter(r['stimulus_set'] for r in sites if r['stimulus_set'])),
                 duplicate_audio_groups=[v for v in groups.values() if len(v)>1],
                 key_sha256=hashlib.sha256((AUTHOR/'KeyforbatsStims.xlsx').read_bytes()).hexdigest())
    write_csv('stimulus_mapping.csv',mappings)
    write_csv('site_stimulus_mapping.csv',assigned)
    write_csv('site_inventory.csv',sites)
    write_csv('wav_inventory.csv',wavs)
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    comparisons = []
    for name in ['boutFMB'+x+'n.wav' for x in 'ABCDEF'] + [r['key_filename'] for r in sets['extended22'] if r['key_category']=='USV'] + ['fourDiffEcho'+x+'n.wav' for x in 'ABCDEF']:
        extended_name = name + '_norm.wav' if name.startswith('voc trial') else name.replace('fourDiffEcho','fourDiff')
        paths = [audio_root/'Vocalization'/name, audio_root/'VocalizationExtended'/extended_name]
        arrays=[]
        for p in paths:
            with wave.open(str(p),'rb') as w:
                if w.getsampwidth()!=2 or w.getnchannels()!=1:
                    raise ValueError(f'Comparison expects mono PCM16: {p}')
                arrays.append(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float))
        x,y=arrays
        a,b=[by_path[p.relative_to(ROOT).as_posix()] for p in paths]
        same_grid = a['sample_rate_hz']==b['sample_rate_hz'] and x.shape==y.shape
        comparisons.append(dict(vocalization_file=name,extended_file=extended_name,
            same_sample_grid=same_grid,identical_pcm=same_grid and np.array_equal(x,y),
            pearson_r=float(np.corrcoef(x,y)[0,1]) if same_grid else '',
            interpretation='Correlation is descriptive, not proof of identical playback or provenance.'))
    write_csv('vocal_extended_comparison.csv',comparisons)
    print(json.dumps({k:v for k,v in summary.items() if k!='duplicate_audio_groups'},indent=2))
    if errors: raise SystemExit('Category mismatches: inspect audit_summary.json')


if __name__=='__main__':
    main()
