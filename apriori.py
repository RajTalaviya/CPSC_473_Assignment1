"""Level-wise Apriori with prefix joins and bitset support counting."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from itertools import combinations, groupby
from pathlib import Path
import sys
from time import perf_counter


def read_transactions(path):
    transactions = []
    seen_ids = set()
    with open(path, encoding='utf-8-sig') as source:
        first = source.readline().strip()
        try:
            declared = int(first)
        except ValueError:
            raise ValueError('First line must be a transaction count.') from None
        if declared <= 0:
            raise ValueError('Transaction count must be positive.')
        for line_number, line in enumerate(source, 2):
            if not line.strip():
                continue
            try:
                values = list(map(int, line.split()))
            except ValueError:
                raise ValueError(f'Line {line_number}: expected integer fields.') from None
            if len(values) < 2:
                raise ValueError(f'Line {line_number}: missing ID or item count.')
            tid, size, *items = values
            if size < 0 or size != len(items):
                raise ValueError(f'Line {line_number}: item count does not match row.')
            if tid in seen_ids:
                raise ValueError(f'Line {line_number}: duplicate transaction ID.')
            if len(set(items)) != len(items):
                raise ValueError(f'Line {line_number}: repeated item in transaction.')
            seen_ids.add(tid)
            transactions.append(frozenset(items))
    if declared != len(transactions):
        raise ValueError(f'Header declares {declared} transactions; found {len(transactions)}.')
    return transactions


def support_threshold(percent, count):
    try:
        value = Decimal(str(percent))
    except InvalidOperation:
        raise ValueError('Support must be a number in (0, 100].') from None
    if not value.is_finite() or not 0 < value <= 100:
        raise ValueError('Support must be a number in (0, 100].')
    return int((value * count / 100).to_integral_value(rounding=ROUND_CEILING))


def generate_candidates(level):
    # Only itemsets sharing their first k-1 items can join.
    ordered = sorted(level)
    for _, members in groupby(ordered, key=lambda itemset: itemset[:-1]):
        for left, right in combinations(list(members), 2):
            candidate = left + (right[-1],)
            # The last two deletions are the parents, already known frequent.
            if all(candidate[:i] + candidate[i + 1:] in level
                   for i in range(len(candidate) - 2)):
                yield candidate, left, right


def apriori(transactions, minimum):
    counts = Counter(item for row in transactions for item in row)
    frequent_items = {item for item, count in counts.items() if count >= minimum}
    # Pack transaction membership into bytes first, avoiding repeated large-int copies.
    buffers = {item: bytearray((len(transactions) + 7) // 8)
               for item in frequent_items}
    for tid, row in enumerate(transactions):
        for item in row & frequent_items:
            buffers[item][tid // 8] |= 1 << (tid % 8)
    level = {(item,): int.from_bytes(bits, 'little')
             for item, bits in buffers.items()}
    del buffers
    results = {(item,): counts[item] for item in frequent_items}
    while level:
        next_level = {}
        for candidate, left, right in generate_candidates(level):
            bits = level[left] & level[right]
            count = bits.bit_count()
            if count >= minimum:
                next_level[candidate] = bits
                results[candidate] = count
        level = next_level
    return results


def write_results(path, patterns):
    with open(path, 'w', encoding='utf-8', newline='\n') as output:
        output.write(f'|FPs| = {len(patterns)}\n')
        for items in sorted(patterns, key=lambda x: (len(x), x)):
            output.write(f'{", ".join(map(str, items))} : {patterns[items]}\n')


def main():
    parser = argparse.ArgumentParser(description='Mine all frequent itemsets using Apriori.')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('minsup', help='Minimum support percentage, e.g. 50 or 2.5')
    args = parser.parse_args()
    start = perf_counter()
    try:
        # Validate the percentage before reading a potentially large file.
        support_threshold(args.minsup, 1)
        transactions = read_transactions(args.dataset)
        minimum = support_threshold(args.minsup, len(transactions))
        patterns = apriori(transactions, minimum)
        output = Path(f'mining_result_{args.dataset.stem}.txt')
        write_results(output, patterns)
    except (OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    elapsed = perf_counter() - start
    print(f'minsup = {args.minsup}% = {minimum}')
    print(f'|FPs| = {len(patterns)}')
    print(f'Total Runtime: {elapsed:.6f} sec')
    print(f'Output: {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
