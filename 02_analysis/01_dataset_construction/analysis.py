import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / '01_data' / 'cpsc_recalls_all.json'
APPAREL_KW = re.compile('\\b(pajama|sleepwear|nightgown|night gown|nightshirt|robe|loungewear|hoodie|sweatshirt|sweater|jacket|coat|parka|vest|shirt|t-shirt|tee shirt|blouse|pants|trousers|jeans|shorts|skirt|dress(es)?|gown|romper|onesie|bodysuit|jumpsuit|overalls|swimwear|swimsuit|bikini|underwear|bra|panties|socks?|hosiery|tights|leggings|scarf|scarves|gloves?|mittens?|hats?|beanie|costume|slippers?|footwear|shoes?|boots?|sandals?|sneakers)\\b')
HOME_KW = re.compile('\\b(blanket|comforter|quilt|bedding|sheet set|pillowcase|curtain|drape|towel|bathrobe|bath robe|mattress pad|throw blanket)\\b')
EXCLUDE_KW = re.compile('\\b(dresser|chest of drawers|drawer chest|storage unit|armoire|wardrobe cabinet|clothing storage|clothes storage|furniture|bunk bed|crib|stroller|high chair|helmet|life jacket|life vest|tractor|mower|vacuum|compressor|heater|stove|grill|fireplace|bicycle|motorcycle|atv\\b|scooter|battery pack|charger|lidocaine|lotion|ointment|medicine|drug|dietary supplement|candle|lighter|seat belt|saddle|scuba|buoyancy|earmuffs?|hard hats?|ornaments?|water pump|fire suppression|figurine|bassinet|boiler|holiday lights|string lights|extension cords?|beanbag|video game|swimming vest|swim vest|mattress(es)? |play yard|greeting card)\\b')
HAZARD_TAXONOMY = [('flammability_burn', 'flammab|\\bburns?\\b|catch fire|ignit|\\bfire\\b'), ('drawstring_strangulation', 'drawstring|strangul'), ('choking_small_parts', 'chok|small part|asphyx|suffocat|detach.*(snap|button|pom)'), ('chemical', '\\blead\\b|phthalate|benzene|formaldehyde|chemical|toxic|poison'), ('entrapment_entanglement', 'entrap|entangl'), ('laceration_puncture', 'lacerat|puncture|sharp'), ('protective_failure', 'fail(ure)? to (provide|meet).{0,40}protect|impact hazard|fail to protect|compression and impact'), ('fall_slip', '\\bfall(s|ing)?\\b|slip hazard|tripping'), ('other', '.')]

def joined_text(rec, keys=('Title', 'Description')):
    parts = [rec.get(k) or '' for k in keys]
    for p in rec.get('Products') or []:
        parts.append(p.get('Name') or '')
    return ' '.join(parts).lower()

def hazard_text(rec):
    return ' '.join((h.get('Name') or '' for h in rec.get('Hazards') or [])).lower()

def classify_hazard(htext, fulltext):
    for blob in (htext, fulltext):
        if not blob.strip():
            continue
        for label, pat in HAZARD_TAXONOMY:
            if label == 'other':
                break
            if re.search(pat, blob):
                return label
    return 'other'

def parse_units(rec):
    vals = []
    for p in rec.get('Products') or []:
        s = (p.get('NumberOfUnits') or '').lower().replace(',', '')
        m = re.search('([\\d.]+)\\s*million', s)
        if m:
            vals.append(float(m.group(1)) * 1000000)
            continue
        m = re.search('(\\d+)', s)
        if m:
            vals.append(float(m.group(1)))
    return sum(vals) if vals else None

def _num(s):
    try:
        return float(s.rstrip('.'))
    except ValueError:
        return None

def parse_price(retail_text):
    t = retail_text.replace(',', '')
    m = re.search('between \\$([\\d.]+) and \\$([\\d.]+)', t)
    if m:
        lo, hi = (_num(m.group(1)), _num(m.group(2)))
        if lo is not None and hi is not None:
            return (lo + hi) / 2
    m = re.search('\\$([\\d.]+)', t)
    if m:
        return _num(m.group(1))
    return None
STORE_KW = re.compile("\\b(store|stores|nationwide|retailers|boutique|outlet|shop|mall|department|costco|walmart|target|kohl|macy|nordstrom|tj ?maxx|ross|marshalls|burlington|meijer|sears|kmart|jcpenney|sam's club|bj's)\\b", re.I)
ONLINE_KW = re.compile('(online|\\.com|\\.net|website|internet|e-?commerce)', re.I)

def sales_channel(retail_text):
    if not retail_text.strip():
        return 'unknown'
    online = bool(ONLINE_KW.search(retail_text))
    store = bool(STORE_KW.search(retail_text))
    if online and (not store):
        return 'online_only'
    if store and (not online):
        return 'store_only'
    if online and store:
        return 'mixed'
    return 'unknown'
CHILD_KW = re.compile('\\b(child|children|kids?|toddler|infant|baby|babies|boys?|girls?|youth|nursery|juvenile)\\b')
FIBER_KW = re.compile('(\\d{1,3})\\s*%\\s*(cotton|polyester|rayon|viscose|nylon|spandex|elastane|wool|acrylic|linen|silk|fleece|modal|bamboo)')

def main():
    with open(RAW, encoding='utf-8-sig') as f:
        raw = json.load(f)
    rows, counts = ([], Counter(total=len(raw)))
    for rec in raw:
        text = joined_text(rec)
        if EXCLUDE_KW.search(text):
            counts['excluded_by_rule'] += 1
            continue
        is_app = bool(APPAREL_KW.search(text))
        is_home = bool(HOME_KW.search(text))
        if not (is_app or is_home):
            counts['not_textile'] += 1
            continue
        counts['included'] += 1
        url_slug = (rec.get('URL') or '').rsplit('/', 1)[-1].replace('-', ' ').lower()
        htext = hazard_text(rec)
        retail = ' '.join((r.get('Name') or '' for r in rec.get('Retailers') or []))
        countries = [c.get('Country') for c in rec.get('ManufacturerCountries') or [] if c.get('Country')]
        injuries = ' '.join((i.get('Name') or '' for i in rec.get('Injuries') or [])).lower()
        remedy_opts = sorted({o.get('Option') for o in rec.get('RemedyOptions') or [] if o.get('Option')})
        fibers = FIBER_KW.findall((rec.get('Description') or '').lower())
        rows.append({'recall_id': rec.get('RecallID'), 'recall_number': rec.get('RecallNumber'), 'recall_date': (rec.get('RecallDate') or '')[:10], 'year': int(rec['RecallDate'][:4]) if rec.get('RecallDate') else None, 'title': rec.get('Title'), 'segment': 'apparel' if is_app else 'home_textile', 'is_childrens': bool(CHILD_KW.search(text)), 'hazard_category': classify_hazard(htext, text + ' ' + url_slug), 'is_violation': 'violat' in htext + text + url_slug or 'flammable fabrics' in text + ' ' + url_slug, 'injuries_reported': not ('none report' in injuries or 'no injuries' in injuries or injuries.strip() == ''), 'units': parse_units(rec), 'price_usd': parse_price(retail), 'sales_channel': sales_channel(retail), 'n_countries': len(countries), 'countries': ';'.join(countries), 'primary_country': countries[0] if countries else None, 'remedy_options': ';'.join(remedy_opts), 'main_fiber': fibers[0][1] if fibers else None, 'url': rec.get('URL')})
    df = pd.DataFrame(rows).sort_values('recall_date').reset_index(drop=True)
    df.to_csv(HERE / 'apparel_recalls_clean.csv', index=False)
    lines = ['# Inclusion summary: apparel and home textile recalls', '', '| Step | Count |', '|---|---|', f"| Raw CPSC recalls (1973-2026) | {counts['total']} |", f"| Excluded: non-textile products | {counts['not_textile']} |", f"| Excluded: exclusion keywords (furniture, gear, etc.) | {counts['excluded_by_rule']} |", f"| **Included in analysis dataset** | **{counts['included']}** |", f"| ... of which apparel | {(df.segment == 'apparel').sum()} |", f"| ... of which home textile | {(df.segment == 'home_textile').sum()} |", f'| ... children-related | {df.is_childrens.sum()} |', f'| ... with country of manufacture | {df.primary_country.notna().sum()} |', f'| ... with parsable units | {df.units.notna().sum()} |', f'| ... with parsable price | {df.price_usd.notna().sum()} |']
    (HERE / 'table_inclusion_summary.md').write_text('\n'.join(lines) + '\n')
    print(df.shape)
    print(df.hazard_category.value_counts())
if __name__ == '__main__':
    main()
